import asyncio
import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, ValidationError
from aiokafka import AIOKafkaProducer
import aiomqtt
import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hardware_gateway")

# Environment configurations
MQTT_BROKER  = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT    = int(os.getenv("MQTT_PORT", 1883))
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC", "raw-flows")

# Redis for marking physical devices as online
REDIS_HOST   = os.getenv("REDIS_HOST", "redis")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

# Topics to subscribe to:
#  devicedna/flows/#    → full flow payloads from capable physical nodes
#  devicedna/telemetry  → ESP32 heartbeat/telemetry (esp32_relay.ino publishes here)
#  devicedna/status/#   → registration/status pings
MQTT_TOPICS = [
    "devicedna/flows/#",
    "devicedna/telemetry",
    "devicedna/status/#",
]

# ── Known physical device classes ─────────────────────────────────────────────
PHYSICAL_DEVICE_CLASS = {
    "dht11_sensor":   "sensor",
    "mq135_sensor":   "sensor",
    "ir_sensor":      "sensor",
    "ldr_sensor":     "sensor",
    "smoke_sensor_1": "sensor",
    "smoke_sensor_2": "sensor",
    "gyro_sensor":    "sensor",
    "esp8266_wifi":   "sensor",
    "HW-001": "sensor",
    "HW-002": "sensor",
    "HW-003": "access_control",
    "HW-004": "industrial",
    "HW-005": "sensor",
}

# ── Strict schema for full flow payloads ──────────────────────────────────────
class FlowSchema(BaseModel):
    flow_id:      str
    src_ip:       str
    dst_ip:       str
    src_port:     int
    dst_port:     int
    protocol:     str
    bytes:        int
    packets:      int
    duration:     float
    timestamp:    str
    device_id:    str
    device_class: str


def mark_device_online(device_id: str) -> None:
    """Refresh the Redis registry key so the dashboard shows ONLINE."""
    try:
        d_class  = PHYSICAL_DEVICE_CLASS.get(device_id, "sensor")
        now_iso  = datetime.now(timezone.utc).isoformat()
        registry_data = {
            "device_id":    device_id,
            "device_class": d_class,
            "source":       "physical",
            "last_seen":    now_iso,
            "status":       "online",
        }
        redis_client.set(f"registry:{device_id}", json.dumps(registry_data))
        logger.info(f"Registry updated: {device_id} -> ONLINE")
    except Exception as e:
        logger.error(f"Failed to mark {device_id} online in Redis: {e}")


def build_synthetic_flow(data: dict) -> dict:
    """
    Convert an ESP32 heartbeat payload (devicedna/telemetry) into a minimal
    flow record that can be forwarded to Kafka for trust scoring.
    """
    device_id = data.get("device_id", "unknown")
    d_class   = PHYSICAL_DEVICE_CLASS.get(device_id, "sensor")
    now_iso   = datetime.now(timezone.utc).isoformat() + "Z"
    return {
        "flow_id":         str(uuid.uuid4()),
        "device_id":       device_id,
        "device_class":    d_class,
        "src_ip":          "192.168.1.100",
        "dst_ip":          "192.168.1.1",
        "src_port":        random.randint(49152, 65535),
        "dst_port":        1883,
        "protocol":        "MQTT",
        "bytes":           int(data.get("total_bytes", 512)),
        "packets":         int(data.get("total_flows", 10)),
        "duration":        5.0,
        "timestamp":       now_iso,
        "total_flows":     int(data.get("total_flows", 10)),
        "total_bytes":     int(data.get("total_bytes", 512)),
        "avg_packet_size": int(data.get("avg_packet_size", 128)),
        "external_ratio":  float(data.get("external_ratio", 0.05)),
        "https_ratio":     float(data.get("https_ratio", 0.5)),
        "tcp_ratio":       float(data.get("tcp_ratio", 0.9)),
        "unique_dst_ips":  int(data.get("unique_dst_ips", 2)),
        "unique_dst_ports":int(data.get("unique_dst_ports", 2)),
    }


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
    producer = await init_kafka_producer()

    reconnect_interval = 5
    while True:
        try:
            logger.info(f"Attempting connection to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
            async with aiomqtt.Client(hostname=MQTT_BROKER, port=MQTT_PORT) as client:

                for topic in MQTT_TOPICS:
                    await client.subscribe(topic)
                    logger.info(f"Subscribed to topic: {topic}")

                logger.info("Listening for physical device telemetry on all topics...")

                async for message in client.messages:
                    topic_str   = str(message.topic)
                    payload_str = ""
                    try:
                        payload_str = message.payload.decode('utf-8')
                        data        = json.loads(payload_str)
                        device_id   = data.get("device_id")

                        # ── Status/registration pings (devicedna/status/#) ──────
                        if topic_str.startswith("devicedna/status/"):
                            if device_id:
                                mark_device_online(device_id)
                                logger.info(f"Status ping from {device_id} — marked ONLINE")
                            continue

                        # ── ESP32 telemetry heartbeat (devicedna/telemetry) ──────
                        if topic_str == "devicedna/telemetry":
                            if device_id:
                                mark_device_online(device_id)
                                flow = build_synthetic_flow(data)
                                await producer.send_and_wait(KAFKA_TOPIC, flow)
                                logger.info(f"ESP32 heartbeat -> Kafka flow for {device_id}")
                            else:
                                logger.warning(f"Telemetry missing device_id: {payload_str[:80]}")
                            continue

                        # ── Full flow payload (devicedna/flows/#) ────────────────
                        try:
                            flow = FlowSchema(**data)
                            await producer.send_and_wait(KAFKA_TOPIC, flow.model_dump())
                            mark_device_online(flow.device_id)
                            logger.info(f"Forwarded full flow {flow.flow_id} from {flow.device_id} -> Kafka")
                        except ValidationError as e:
                            errors = [{"field": err["loc"][0], "msg": err["msg"]} for err in e.errors()]
                            logger.warning(f"Schema validation failed: {errors}")

                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON payload on {topic_str}: {message.payload[:80]}")
                    except Exception as e:
                        logger.error(f"Error processing message on {topic_str}: {e}")

        except aiomqtt.MqttError as e:
            logger.warning(f"MQTT connection error: {e}. Reconnecting in {reconnect_interval}s...")
            await asyncio.sleep(reconnect_interval)
        except Exception as e:
            logger.error(f"Gateway critical error: {e}. Restarting in {reconnect_interval}s...")
            await asyncio.sleep(reconnect_interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Gateway bridge stopped by user.")
