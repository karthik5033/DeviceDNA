import asyncio
import json
import logging
import os
from pydantic import BaseModel, ValidationError
from aiokafka import AIOKafkaProducer
import aiomqtt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hardware_gateway")

# Environment configurations
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "devicedna/flows/#")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw-flows")

# Pydantic schema for strict payload validation
class FlowSchema(BaseModel):
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes: int
    packets: int
    duration: float
    timestamp: str
    device_id: str
    device_class: str

async def init_kafka_producer() -> AIOKafkaProducer:
    """Initialize Kafka producer with reconnect logic."""
    while True:
        try:
            producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await producer.start()
            logger.info(f"Successfully connected to Kafka at {KAFKA_BROKER}")
            return producer
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

async def main():
    logger.info("Starting DeviceDNA Hardware Gateway Bridge (MQTT -> Kafka)...")
    
    # Wait for Kafka to be ready before consuming MQTT
    producer = await init_kafka_producer()

    reconnect_interval = 5
    while True:
        try:
            logger.info(f"Attempting connection to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
            
            # Using aiomqtt's context manager which automatically handles the connection
            async with aiomqtt.Client(hostname=MQTT_BROKER, port=MQTT_PORT) as client:
                logger.info(f"Connected to MQTT broker. Subscribing to topic: {MQTT_TOPIC}")
                await client.subscribe(MQTT_TOPIC)
                logger.info("Subscription active. Listening for incoming physical telemetry...")
                
                # Consume messages asynchronously
                async for message in client.messages:
                    try:
                        payload_str = message.payload.decode('utf-8')
                        data = json.loads(payload_str)
                        
                        # Validate against our defined schema
                        flow = FlowSchema(**data)
                        
                        # Forward valid payload to Kafka topic
                        await producer.send_and_wait(KAFKA_TOPIC, flow.model_dump())
                        logger.info(f"Forwarded valid flow {flow.flow_id} from physical device {flow.device_id} -> Kafka")
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Dropped payload - Invalid JSON format: {message.payload}")
                    except ValidationError as e:
                        # Extract validation errors gracefully
                        errors = [{"field": err["loc"][0], "msg": err["msg"]} for err in e.errors()]
                        logger.warning(f"Dropped payload - Schema validation failed. Reasons: {errors}. Payload: {payload_str}")
                    except Exception as e:
                        logger.error(f"Unexpected error processing message: {e}")
                        
        except aiomqtt.MqttError as e:
            logger.warning(f"MQTT connection error: {e}. Reconnecting in {reconnect_interval} seconds...")
            await asyncio.sleep(reconnect_interval)
        except Exception as e:
            logger.error(f"Gateway experienced unexpected critical error: {e}. Restarting loop in {reconnect_interval} seconds...")
            await asyncio.sleep(reconnect_interval)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Gateway bridge stopped by user.")
