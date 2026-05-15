import asyncio
import json
import logging
import traceback
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from app.api.ws import sio
from app.services.feature_extraction import extract_features
from app.services.trust_engine import master_trust_engine

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
            task = asyncio.create_task(self._consume())
            
            def _handle_task_result(t):
                try:
                    t.result()
                except Exception as ex:
                    import traceback
                    logger.error(f"Task crashed unhandled:\n{traceback.format_exc()}")
                    
            task.add_done_callback(_handle_task_result)
        except Exception as e:
            logger.error(f"Kafka connection failed (Will not stream telemetry): {e}")

    async def _consume(self):
        async for msg in self.consumer:
            flow = msg.value
            logger.info(f"Received flow from Kafka: {flow.get('flow_id', 'unknown')}")
            await self._process_flow(flow)

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

        # ML Pipeline: extract features, evaluate trust, and broadcast score
        # ML Pipeline: extract features, evaluate trust, and broadcast score
        features = extract_features(
            flow.get('device_id', 'unknown'),
            flow.get('device_class', 'unknown'),
            [flow]
        )
        trust_score = await master_trust_engine.evaluate_device(
            flow.get('device_id', 'unknown'),
            flow.get('device_class', 'unknown'),
            features.to_tensor_list() if hasattr(features, 'to_tensor_list') else [],
            {}
        )
        await sio.emit('trust_update', {
            'device_id': flow.get('device_id'),
            'score': trust_score,
            'timestamp': datetime.utcnow().isoformat()
        })

    async def stop(self):
        if self.consumer:
            logger.info("Stopping TelemetryService...")
            await self.consumer.stop()
