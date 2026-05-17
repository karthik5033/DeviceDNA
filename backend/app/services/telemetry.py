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
                group_id="backend_telemetry",
                auto_offset_reset="latest",
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            await self.consumer.start()
            logger.info(f"TelemetryService: Listening to {RAW_TOPIC} on {KAFKA_BROKER}")
            with open('debug_telemetry.log', 'a') as f: f.write("TelemetryService started\n")
            task = asyncio.create_task(self._consume())
            
            def _handle_task_result(t):
                try:
                    t.result()
                except Exception as ex:
                    import traceback
                    logger.error(f"Task crashed unhandled:\n{traceback.format_exc()}")
                    with open('debug_telemetry.log', 'a') as f: f.write(f"Task crashed: {traceback.format_exc()}\n")
                    
            task.add_done_callback(_handle_task_result)
        except Exception as e:
            logger.error(f"Kafka connection failed (Will not stream telemetry): {e}")
            with open('debug_telemetry.log', 'a') as f: f.write(f"Kafka failed: {e}\n")

    async def _consume(self):
        try:
            with open('debug_telemetry.log', 'a') as f: f.write("Entered _consume loop\n")
            async for msg in self.consumer:
                flow = msg.value
                with open('debug_telemetry.log', 'a') as f: f.write(f"Got flow: {flow.get('flow_id')}\n")
                await self._process_flow(flow)
        except Exception as e:
            import traceback
            with open('debug_telemetry.log', 'a') as f: f.write(f"Consume crash: {traceback.format_exc()}\n")

    async def _process_flow(self, flow):
        try:
            if flow.get('is_anomaly') or "policy_violation" in str(flow).lower():
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
                
            if 'src_ip' in flow:
                await sio.emit('telemetry_ping', {
                    'source': flow.get('src_ip'),
                    'target': flow.get('dst_ip'),
                    'bytes': flow.get('bytes')
                })

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
            
            with open('debug_telemetry.log', 'a') as f: f.write(f"Score for {flow.get('device_id')}: {trust_score}\n")
            
            final_score_value = float(trust_score.get('trust_score', 0.0)) if isinstance(trust_score, dict) else float(trust_score)
            
            payload = {
                'device_id': flow.get('device_id'),
                'score': final_score_value,
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.info(f"EMITTING PAYLOAD: {payload}")
            
            await sio.emit('trust_update', payload)
            with open('debug_telemetry.log', 'a') as f: f.write(f"Emitted trust_update for {flow.get('device_id')}\n")
        except Exception as e:
            import traceback
            with open('debug_telemetry.log', 'a') as f: f.write(f"Process crash: {traceback.format_exc()}\n")

    async def stop(self):
        if self.consumer:
            logger.info("Stopping TelemetryService...")
            await self.consumer.stop()
