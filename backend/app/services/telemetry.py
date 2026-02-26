import asyncio
import json
import logging
from datetime import datetime
from aiokafka import AIOKafkaConsumer

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
        self.consumer = AIOKafkaConsumer(
            RAW_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        await self.consumer.start()
        logger.info(f"TelemetryService: Listening to {RAW_TOPIC} on {KAFKA_BROKER}")
        asyncio.create_task(self._consume())

    async def _consume(self):
        try:
            async for msg in self.consumer:
                flow = msg.value
                self._process_flow(flow)
        except Exception as e:
            logger.error(f"Error consuming telemetry: {e}")
        finally:
            await self.consumer.stop()

    def _process_flow(self, flow):
        # Normalization and InfluxDB persistence for raw flows
        # (This is a stub, as the InfluxDB client implementation is next)
        if "policy_violation" in str(flow):
            logger.info(f"Received attack flow: {flow['flow_id']} - type: {flow.get('attack_type')}")
        pass
        
    async def stop(self):
        if self.consumer:
            logger.info("Stopping TelemetryService...")
            await self.consumer.stop()
