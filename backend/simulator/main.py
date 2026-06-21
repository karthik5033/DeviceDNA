import asyncio
import json
import logging
import os
from aiokafka import AIOKafkaProducer
from simulator.traffic_generator import generate_batch, ACTIVE_RESTRICTIONS
from simulator.device_profiles import FLEET
from simulator.attack_scenarios import AttackScenarios
from app.db.redis import redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
TOPIC_NAME = "raw-flows"

# ── Simulated ESP32 MQTT Listener ─────────────────────────────────────────────
def run_mqtt_listener():
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(client_id="devicedna_simulator_actuator")
        
        def on_connect(client, userdata, flags, rc):
            logger.info(f"Simulator MQTT Actuator connected with result code {rc}")
            client.subscribe("devicedna/+/command")
            
        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode('utf-8'))
                d_id = payload.get("device_id")
                action = payload.get("action")
                relay_open = payload.get("relay_open", False)
                
                if action == "quarantine" or (action == "isolate" or relay_open):
                    logger.warning(f"🔒 [ESP32 Sim-Hardware] DEVICE {d_id} RELAY GPIO 26: OPENED (Traffic cut-off). Status: QUARANTINED.")
                elif action == "rate_limit":
                    logger.warning(f"⚠️ [ESP32 Sim-Hardware] DEVICE {d_id} RELAY GPIO 26: CLOSED. Status: RATE LIMITED (Throttling traffic).")
                elif action == "recover":
                    logger.info(f"🟢 [ESP32 Sim-Hardware] DEVICE {d_id} RELAY GPIO 26: CLOSED. Status: RECOVERED & RESTORED.")
            except Exception as e:
                pass
                
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_HOST, 1883, 60)
        client.loop_start()
        return client
    except Exception as e:
        logger.warning(f"Could not start MQTT actuator listener (paho-mqtt missing or broker offline): {e}")
        return None

async def stream_telemetry():
    """
    Simulate a constant stream of IoT network telemetry.
    Starts generating normal batches of 100 flows every second, producing them to Kafka.
    """
    logger.info(f"Initializing DeviceDNA Telemetry Simulator targeting {KAFKA_BROKER}...")
    
    # Start the background MQTT subscriber
    mqtt_client = run_mqtt_listener()
    
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Retry logic if Kafka isn't up yet
    retries = 5
    while retries > 0:
        try:
            await producer.start()
            logger.info("Successfully connected to Kafka.")
            break
        except Exception as e:
            logger.warning(f"Waiting for Kafka ({retries} retries left): {e}")
            await asyncio.sleep(5)
            retries -= 1
            
    if retries == 0:
        logger.error("Could not connect to Kafka. Exiting simulation.")
        return

    try:
        logger.info("Starting baseline telemetry stream...")
        cycle_count = 0
        
        while True:
            cycle_count += 1
            
            # Sync active restrictions from Redis
            try:
                for d in FLEET:
                    d_id = d['id']
                    ACTIVE_RESTRICTIONS[d_id] = {
                        "isolated": redis_client.exists(f"response:isolated:{d_id}") == 1,
                        "rate_limited": redis_client.exists(f"response:rate_limit:{d_id}") == 1,
                        "sandboxed": redis_client.exists(f"response:sandboxed:{d_id}") == 1,
                        "honeypot": redis_client.exists(f"response:honeypot:{d_id}") == 1,
                    }
            except Exception as redis_err:
                logger.error(f"Failed to sync restrictions from Redis: {redis_err}")
            
            # Generate 100 normal flows
            flows = generate_batch(100)
            
            # Scenario Injection (Random chance every ~100 cycles)
            if cycle_count % 100 == 0 and os.getenv("DISABLE_ATTACK_INJECTION", "false").lower() != "true":
                logger.info(f"Cycle {cycle_count}: Injecting Threat Scenarios...")
                flows.append(AttackScenarios.scenario_1_botnet_c2())
                flows.append(AttackScenarios.scenario_2_slow_exfiltration())
                flows.append(AttackScenarios.scenario_3_lateral_movement())
                flows.append(AttackScenarios.scenario_4_nlp_policy_trigger())

            for flow in flows:
                await producer.send_and_wait(TOPIC_NAME, flow)
                
            if cycle_count % 10 == 0:
                logger.info(f"Streamed {cycle_count * 100} flows to topic: {TOPIC_NAME}...")
                
            await asyncio.sleep(0.5)  # Pace the simulation
            
    except asyncio.CancelledError:
        logger.info("Telemetry streaming cancelled.")
    except KeyboardInterrupt:
        logger.info("Simulator halted.")
    finally:
        await producer.stop()
        if mqtt_client:
            mqtt_client.loop_stop()
        logger.info("Kafka producer stopped.")

if __name__ == "__main__":
    asyncio.run(stream_telemetry())
