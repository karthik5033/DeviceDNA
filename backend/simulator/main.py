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

# Global set of active compromised simulator devices and their configurations
# Format: { device_id: { "attacker_ip": str, "attack_type": str, "start_time": float } }
COMPROMISED_DEVICES = {}

async def run_exploit_listener():
    """
    Listens for exploit trigger packets from the Attacker Laptop on port 8888.
    """
    class ExploitServerProtocol(asyncio.Protocol):
        def connection_made(self, transport):
            self.transport = transport

        def data_received(self, data):
            try:
                message = data.decode('utf-8').strip()
                # Format: EXPLOIT:<device_id>:<attacker_ip>:<recon|beacon|lateral|exfil|coordinated>
                if message.startswith("EXPLOIT:"):
                    parts = message.split(":")
                    if len(parts) >= 4:
                        _, device_id, attacker_ip, attack_type = parts
                        logger.warning(f"💥 [Simulator Vulnerability] EXPLOIT RECEIVED! Target: {device_id} | Attacker: {attacker_ip} | Attack: {attack_type}")
                        import time
                        COMPROMISED_DEVICES[device_id] = {
                            "attacker_ip": attacker_ip,
                            "attack_type": attack_type,
                            "start_time": time.time()
                        }
                        try:
                            redis_client.setex(f"compromised:{device_id}", 3600, json.dumps({
                                "attacker_ip": attacker_ip,
                                "attack_type": attack_type,
                                "start_time": time.time()
                            }))
                        except Exception as redis_err:
                            logger.error(f"Failed to write compromised state to Redis: {redis_err}")
                self.transport.close()
            except Exception as e:
                logger.error(f"Error processing exploit trigger: {e}")

    loop = asyncio.get_running_loop()
    server = await loop.create_server(ExploitServerProtocol, '0.0.0.0', 8888)
    logger.info("Simulator Exploit Listener running on port 8888...")
    async with server:
        await server.serve_forever()

async def run_compromised_device_logic():
    """
    Executes real outbound network connections for compromised devices.
    - Beaconing: periodic TCP connections to Attacker C2 on port 4444.
    - Exfiltration: periodic TCP connections to Attacker Exfil on port 9999.
    - Lateral: TCP socket probes to peer device IPs.
    """
    import time
    beacon_timestamps = {}
    exfil_timestamps = {}
    lateral_timestamps = {}
    exfil_stages = {}

    while True:
        now = time.time()
        for device_id, comp in list(COMPROMISED_DEVICES.items()):
            attacker_ip = comp["attacker_ip"]
            attack_type = comp["attack_type"]
            device = next((d for d in FLEET if d["id"] == device_id), None)
            if not device:
                continue

            # 1. Botnet Beaconing (every 30 seconds)
            if attack_type in ["beacon", "coordinated"]:
                last_beacon = beacon_timestamps.get(device_id, 0.0)
                if now - last_beacon >= 30:
                    beacon_timestamps[device_id] = now
                    asyncio.create_task(send_socket_payload(
                        host=attacker_ip,
                        port=4444,
                        payload=f"BEACON:{device_id}:ALIVE\n"
                    ))

            # 2. Slow Exfiltration (every 60 seconds)
            if attack_type in ["exfil", "coordinated"]:
                last_exfil = exfil_timestamps.get(device_id, 0.0)
                if now - last_exfil >= 60:
                    exfil_timestamps[device_id] = now
                    stage = exfil_stages.get(device_id, 0)
                    sizes = [500, 600, 800, 1200, 2000, 3500]
                    current_size = sizes[min(stage, len(sizes)-1)]
                    exfil_stages[device_id] = stage + 1
                    
                    asyncio.create_task(send_socket_payload(
                        host=attacker_ip,
                        port=9999,
                        payload="A" * current_size
                    ))

            # 3. Lateral Movement (every 10 seconds)
            if attack_type in ["lateral", "coordinated"]:
                last_lateral = lateral_timestamps.get(device_id, 0.0)
                if now - last_lateral >= 10:
                    lateral_timestamps[device_id] = now
                    # Select a random peer IP
                    import random
                    peers = [d for d in FLEET if d["id"] != device_id]
                    if peers:
                        target_peer = random.choice(peers)
                        port = random.choice([22, 5432, 1883, 80])
                        asyncio.create_task(probe_socket_port(
                            host=target_peer["ip_address"],
                            port=port
                        ))

        await asyncio.sleep(1)

async def send_socket_payload(host, port, payload):
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(payload.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        logger.info(f"Outbound payload sent to {host}:{port}")
    except Exception as e:
        logger.debug(f"Failed to connect to attacker receiver at {host}:{port}: {e}")

async def probe_socket_port(host, port):
    try:
        # 1.0s timeout to generate a fast SYN attempt
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=1.0)
        writer.close()
        await writer.wait_closed()
    except Exception:
        # Connection will likely fail, which is expected (still generates SYN packet)
        pass

async def stream_telemetry():
    """
    Simulate a constant stream of IoT network telemetry.
    Starts generating normal batches of 100 flows every second, producing them to Kafka.
    """
    logger.info(f"Initializing DeviceDNA Telemetry Simulator targeting {KAFKA_BROKER}...")
    
    # Start the background MQTT subscriber
    mqtt_client = run_mqtt_listener()
    
    # Start background exploit listener and compromised loops
    asyncio.create_task(run_exploit_listener())
    asyncio.create_task(run_compromised_device_logic())
    
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
