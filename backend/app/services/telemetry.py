import asyncio
import json
import logging
import math
import time
import traceback
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from app.api.ws import sio
from app.services.feature_extraction import extract_features
from app.services.trust_engine import master_trust_engine
from app.services.hardware_registry import mark_seen
from app.ml.gnn.scoring import gnn_scorer
from app.db.redis import redis_client
from simulator.device_profiles import FLEET

IP_TO_DEVICE_ID = {d['ip_address']: d['id'] for d in FLEET}

logger = logging.getLogger(__name__)

import os
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
RAW_TOPIC = "raw-flows"

# Minimum seconds between trust evaluations per device (time-based fallback)
MIN_EVAL_INTERVAL_SECS = 60

# How many historical scores to use for baseline computation (sliding window)
BASELINE_HISTORY_KEY = "baseline:{device_id}"
BASELINE_WINDOW = 20  # last 20 readings


def _compute_baseline_stats(device_id: str) -> dict:
    """
    Reads the last BASELINE_WINDOW feature snapshots stored in Redis for this device
    and computes mean/std for the three CUSUM-monitored features.
    Returns a dict compatible with CUSUMDriftEngine.detect_drift().
    If insufficient history, returns sensible defaults so CUSUM still runs.
    """
    key = f"baseline:{device_id}"
    raw_entries = redis_client.lrange(key, 0, BASELINE_WINDOW - 1)

    parsed = []
    for entry in raw_entries:
        try:
            parsed.append(json.loads(entry))
        except Exception:
            pass

    target_keys = ["total_bytes", "avg_packet_size", "external_traffic_ratio"]
    stats = {}

    for k in target_keys:
        values = [p[k] for p in parsed if k in p]
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 1.0
        else:
            # Reasonable defaults per feature when history is insufficient
            defaults = {
                "total_bytes": {"mean": 50000.0, "std": 20000.0},
                "avg_packet_size": {"mean": 512.0, "std": 256.0},
                "external_traffic_ratio": {"mean": 0.1, "std": 0.05},
            }
            mean = defaults[k]["mean"]
            std = defaults[k]["std"]
        stats[k] = {"mean": mean, "std": std}

    return stats


def _store_feature_snapshot(device_id: str, features_obj) -> None:
    """
    Push a compact feature snapshot into the per-device Redis list
    used for baseline computation. Caps list length at BASELINE_WINDOW.
    """
    try:
        key = f"baseline:{device_id}"
        snapshot = {
            "total_bytes": float(features_obj.total_bytes),
            "avg_packet_size": float(features_obj.avg_packet_size),
            "external_traffic_ratio": float(features_obj.external_traffic_ratio),
        }
        redis_client.lpush(key, json.dumps(snapshot))
        redis_client.ltrim(key, 0, BASELINE_WINDOW - 1)
        redis_client.expire(key, 86400)  # 24h TTL
    except Exception as e:
        logger.warning(f"Failed to store feature snapshot for baseline [{device_id}]: {e}")

class TelemetryService:
    """
    Consumes raw flows from Kafka, normalizes them, and writes to InfluxDB.
    Runs continuously in the background of the FastAPI app.
    """
    def __init__(self, influx_client):
        self.influx_client = influx_client
        self.consumer = None
        self.flow_count = 0
        # Issue 3: track last trust evaluation timestamp per device
        self._last_eval_time: dict[str, float] = {}

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
                
            is_anom = flow.get('is_anomaly') or flow.get('is_anomalous') or False
            if is_anom:
                master_trust_engine.register_active_attack(device_id, flow.get('attack_type'))

            if is_anom or "policy_violation" in str(flow).lower():
                await sio.emit('new_alert', {
                    'id': flow.get('flow_id', 'ALT-LIVE'),
                    'device': flow.get('device_id', 'Unknown'),
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
            device_id = flow.get('device_id', 'unknown')

            # Issue 3: evaluate on every 10th flow OR if ≥60s have passed since last eval
            now = time.time()
            last_eval = self._last_eval_time.get(device_id, 0.0)
            time_elapsed = (now - last_eval) >= MIN_EVAL_INTERVAL_SECS
            flow_threshold = (self.flow_count % 10 == 0)

            if not flow_threshold and not time_elapsed and not is_anom:
                return

            self._last_eval_time[device_id] = now

            features = extract_features(
                device_id,
                flow.get('device_class', 'unknown'),
                [flow]
            )

            # Issue 1: store snapshot and compute real baseline stats for CUSUM
            _store_feature_snapshot(device_id, features)
            baseline_stats = _compute_baseline_stats(device_id)

            trust_score = await master_trust_engine.evaluate_device(
                device_id,
                flow.get('device_class', 'unknown'),
                features.to_tensor_list() if hasattr(features, 'to_tensor_list') else [],
                baseline_stats
            )
            
            final_score_value = float(trust_score.get('trust_score', 0.0)) if isinstance(trust_score, dict) else float(trust_score)
            
            payload = {
                'device_id': device_id,
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
