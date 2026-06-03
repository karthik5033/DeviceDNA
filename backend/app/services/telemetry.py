import asyncio
import json
import logging
import traceback
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from app.api.ws import sio
from app.services.feature_extraction import extract_features
from app.services.trust_engine import master_trust_engine
from app.services.hardware_registry import mark_seen
from app.ml.gnn.scoring import gnn_scorer
from simulator.device_profiles import FLEET

IP_TO_DEVICE_ID = {d['ip_address']: d['id'] for d in FLEET}

logger = logging.getLogger(__name__)

import os
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
RAW_TOPIC = "raw-flows"

class TelemetryService:
    """
    Consumes raw flows from Kafka, normalizes them, and writes to InfluxDB.
    Runs continuously in the background of the FastAPI app.
    """
    def __init__(self, influx_client):
        self.influx_client = influx_client
        self.consumer = None
        self.flow_count = 0

    async def start(self):
        # Graceful handling if Kafka is not yet up in Docker
        try:
            self.consumer = AIOKafkaConsumer(
                RAW_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id="backend_telemetry_2",
                auto_offset_reset="latest",
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
        try:
            async for msg in self.consumer:
                flow = msg.value
                await self._process_flow(flow)
                # Yield control to the event loop so FastAPI can handle HTTP requests
                await asyncio.sleep(0)
        except Exception as e:
            import traceback
            logger.error(f"Consume crash: {traceback.format_exc()}")

    async def _process_flow(self, flow):
        try:
            device_id = flow.get('device_id')
            device_class = flow.get('device_class')
            
            # Register device as online
            if device_id:
                # We offload it or await it directly
                # It does an async redis set, so awaiting is fine
                await mark_seen(device_id, device_class)
                
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
                src_ip = flow.get('src_ip')
                dst_ip = flow.get('dst_ip')
                
                # Update frontend visualization map
                await sio.emit('telemetry_ping', {
                    'source': src_ip,
                    'target': dst_ip,
                    'bytes': flow.get('bytes')
                })
                
                # Build real GNN edges based on observed communication
                src_id = flow.get('device_id') or IP_TO_DEVICE_ID.get(src_ip)
                dst_id = IP_TO_DEVICE_ID.get(dst_ip)
                if src_id and dst_id:
                    gnn_scorer.update_graph(src_id, dst_id)

            self.flow_count += 1
            if self.flow_count % 10 != 0:
                return
                
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
            
            final_score_value = float(trust_score.get('trust_score', 0.0)) if isinstance(trust_score, dict) else float(trust_score)
            
            payload = {
                'device_id': flow.get('device_id'),
                'score': final_score_value,
                'timestamp': datetime.utcnow().isoformat()
            }
            # Only log every 100th flow to avoid spamming the console
            # logger.info(f"EMITTING PAYLOAD: {payload}")
            
            await sio.emit('trust_update', payload)
        except Exception as e:
            import traceback
            logger.error(f"Process crash: {traceback.format_exc()}")

    async def stop(self):
        if self.consumer:
            logger.info("Stopping TelemetryService...")
            await self.consumer.stop()
