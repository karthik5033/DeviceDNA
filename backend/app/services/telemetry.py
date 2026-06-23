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

def get_device_id_by_ip(ip: str) -> str:
    if not ip:
        return None
    static_id = IP_TO_DEVICE_ID.get(ip)
    if static_id:
        return static_id
        
    try:
        # Check Redis for dynamically learned physical device IPs
        for d in FLEET:
            if d.get('is_physical') or d['id'] in ["dht11_sensor", "mq135_sensor", "ir_sensor", "ldr_sensor", "esp8266_wifi"]:
                dev_id = d['id']
                stored_ip = redis_client.get(f"physical_ip:{dev_id}")
                if stored_ip:
                    if isinstance(stored_ip, bytes):
                        stored_ip = stored_ip.decode('utf-8')
                    if stored_ip == ip:
                        return dev_id
    except Exception:
        pass
    return None

logger = logging.getLogger(__name__)

import os
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
RAW_TOPIC = "raw-flows"

# Minimum seconds between trust evaluations per device (time-based fallback)
MIN_EVAL_INTERVAL_SECS = 5

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
        # track last trust evaluation timestamp per device
        self._last_eval_time: dict[str, float] = {}
        # Buffer flows per device for aggregated ML evaluation
        self._flow_buffer: dict[str, list] = {}

    async def start(self):
        # Graceful handling if Kafka is not yet up in Docker
        try:
            import uuid
            self.consumer = AIOKafkaConsumer(
                RAW_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=f"backend_telemetry_{uuid.uuid4().hex[:8]}",
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
                device_id = flow.get('device_id')
                
                # Check lag
                flow_time_str = flow.get('timestamp')
                if flow_time_str:
                    try:
                        flow_time = datetime.fromisoformat(flow_time_str.replace('Z', '+00:00')).timestamp()
                        lag = time.time() - flow_time
                        if lag > 10:
                            logger.warning(f"Telemetry Consumer is lagging by {lag:.2f} seconds!")
                    except Exception:
                        pass

                if device_id == 'esp8266_wifi':
                    logger.info("Kafka consumer ACTUALLY RECEIVED esp8266_wifi flow!")
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
            
            # Persist raw flow to InfluxDB
            if device_id:
                await self.influx_client.write_flow(flow)
            
            # Register device as online
            if device_id:
                # We offload it or await it directly
                # It does an async redis set, so awaiting is fine
                await mark_seen(device_id, device_class)
                
            # Policy checks on raw flow (e.g., checking if destination IP matches blocklists)
            is_policy_violation = "policy_violation" in str(flow).lower()
            if is_policy_violation:
                await sio.emit('new_alert', {
                    'id': flow.get('flow_id', 'ALT-LIVE'),
                    'device': flow.get('device_id', 'Unknown'),
                    'severity': 'critical',
                    'type': 'policy_violation',
                    'message': f"Live Policy Violation: unauthorized connection attempt to {flow.get('dst_ip')}",
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
                src_id = flow.get('device_id') or get_device_id_by_ip(src_ip)
                dst_id = get_device_id_by_ip(dst_ip)
                if src_id and dst_id:
                    gnn_scorer.update_graph(src_id, dst_id)

            self.flow_count += 1
            device_id = flow.get('device_id', 'unknown')
            
            if device_id == 'esp8266_wifi':
                logger.info(f"ESP8266: Flow received. Current buffer len: {len(self._flow_buffer.get('esp8266_wifi', []))}")

            if device_id not in self._flow_buffer:
                self._flow_buffer[device_id] = []
            self._flow_buffer[device_id].append(flow)

            now = time.time()
            last_eval = self._last_eval_time.get(device_id, 0.0)
            time_elapsed = (now - last_eval) >= MIN_EVAL_INTERVAL_SECS
            
            # Evaluate when we have accumulated a batch of exactly 25 flows per device to match model norms
            flow_threshold = len(self._flow_buffer[device_id]) >= 25

            if device_id == 'esp8266_wifi':
                logger.info(f"ESP8266: flow_threshold={flow_threshold}, time_elapsed={time_elapsed}")

            if not flow_threshold and not time_elapsed and not is_policy_violation:
                return

            if device_id == 'esp8266_wifi':
                logger.info(f"ESP8266: Evaluating now!")

            self._last_eval_time[device_id] = now
            
            # Extract features on the aggregated buffer
            flows_to_eval = self._flow_buffer[device_id]
            self._flow_buffer[device_id] = []

            features = extract_features(
                device_id,
                flow.get('device_class', 'unknown'),
                flows_to_eval
            )

            # Issue 1: store snapshot and compute real baseline stats for CUSUM
            _store_feature_snapshot(device_id, features)
            baseline_stats = _compute_baseline_stats(device_id)
            
            # Persist aggregated 14D feature vector to InfluxDB
            feature_dict = features.__dict__ if hasattr(features, '__dict__') else features
            await self.influx_client.write_feature_vector(device_id, flow.get('device_class', 'unknown'), feature_dict)

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
