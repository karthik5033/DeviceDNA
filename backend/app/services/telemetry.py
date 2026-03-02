import asyncio
import json
import logging
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from app.api.ws import sio

logger = logging.getLogger(__name__)

KAFKA_BROKER = "localhost:29092"
RAW_TOPIC = "raw-flows"

class TelemetryService:
    """
    Consumes raw flows from Kafka, normalizes them, and writes to InfluxDB.
    Runs continuously in the background of the FastAPI app.
    """
    def __init__(self, influx_client):
        self.influx_client = influx_client
        self.consumer = None

    async def start(self):
        # Graceful handling if Kafka is not yet up in Docker
        try:
            self.consumer = AIOKafkaConsumer(
                RAW_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            await self.consumer.start()
            logger.info(f"TelemetryService: Listening to {RAW_TOPIC} on {KAFKA_BROKER}")
            asyncio.create_task(self._consume())
        except Exception as e:
            logger.error(f"Kafka connection failed (Will not stream telemetry): {e}")

    async def _consume(self):
        try:
            async for msg in self.consumer:
                flow = msg.value
                await self._process_flow(flow)
        except Exception as e:
            logger.error(f"Error consuming telemetry: {e}")
        finally:
            if self.consumer:
                await self.consumer.stop()

    async def _process_flow(self, flow):
        # We broadcast some metrics up to the UI if it's an anomaly or a flow ping
        if flow.get('is_anomaly') or "policy_violation" in str(flow).lower():
            logger.info(f"Anomaly Detected! Broadcasting alert: {flow}")
            await sio.emit('new_alert', {
                'id': flow.get('flow_id', 'ALT-LIVE'),
                'device': flow.get('src_ip', 'Unknown'),
                'severity': 'critical',
                'type': flow.get('attack_type', 'Suspicious Activity'),
                'message': f"Live Anomaly Trigger: {flow.get('attack_type')} detected hitting {flow.get('dst_ip')}",
                'score': 25.0,
                'time': 'Just now',
                'model': 'Live Kafka Stream'
            })
            
        # Send a heartbeat telemetry packet to visually "pump" the dashboard network topology
        if 'src_ip' in flow:
            await sio.emit('telemetry_ping', {
                'source': flow.get('src_ip'),
                'target': flow.get('dst_ip'),
                'bytes': flow.get('bytes')
            })

    async def stop(self):
        if self.consumer:
            logger.info("Stopping TelemetryService...")
            await self.consumer.stop()
