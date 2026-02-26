import asyncio
import json
import logging
from aiokafka import AIOKafkaProducer
from simulator.traffic_generator import generate_batch
from simulator.attack_scenarios import AttackScenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BROKER = "localhost:29092"
TOPIC_NAME = "raw-flows"

async def stream_telemetry():
    """
    Simulate a constant stream of IoT network telemetry.
    Starts generating normal batches of 100 flows every second, producing them to Kafka.
    """
    logger.info(f"Initializing DeviceDNA Telemetry Simulator targeting {KAFKA_BROKER}...")
    
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
            
            # Generate 100 normal flows
            flows = generate_batch(100)
            
            # Scenario Injection (Random chance every ~100 cycles)
            if cycle_count % 100 == 0:
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
        logger.info("Kafka producer stopped.")

if __name__ == "__main__":
    asyncio.run(stream_telemetry())
