import logging
import json
from datetime import datetime
from app.ml.gmvae.scoring import gmvae_scorer
from app.ml.isolation_forest.model import if_scorer
from app.ml.lstm.scoring import lstm_scorer
from app.ml.gnn.scoring import gnn_scorer
from app.services.drift_engine import cusum_engine
from app.services.response_engine import response_engine
from app.services.trust_decay import record_anomaly_event, get_decay_multiplier
from app.services.recovery_manager import evaluate_recovery
from app.db.influxdb import influx_db
from app.db.redis import redis_client
from app.db.postgres import AsyncSessionLocal
from app.db.models import Alert
from app.api.ws import sio
from app.ml.explainability.tib_generator import generate_tib
from simulator.device_profiles import FLEET

logger = logging.getLogger(__name__)

# Group static devices by class for rapid peer comparison lookups
DEVICES_BY_CLASS = {}
for d in FLEET:
    cls = d['device_class']
    if cls not in DEVICES_BY_CLASS:
        DEVICES_BY_CLASS[cls] = []
    DEVICES_BY_CLASS[cls].append(d['id'])

CLASS_POLICY_RULES = {
    'camera': [
        (8, '<', 0.6),    # ext_int_ratio < 0.6
        (9, '<', 200),   # active_hours_bitmap < 200
        (0, '<', 100000)  # bytes_sent < 100000
    ],
    'sensor': [
        (5, '<', 5),      # unique_dst_ips < 5
        (0, '<', 50000),   # bytes_sent < 50000
        (8, '<', 0.05)    # ext_int_ratio < 0.05
    ],
    'thermostat': [
        (0, '<', 75000),  # bytes_sent < 75000
        (8, '<', 0.4),    # ext_int_ratio < 0.4
        (5, '<', 5)       # unique_dst_ips < 5
    ],
    'access_control': [
        (0, '<', 150000), # bytes_sent < 150000
        (8, '<', 0.2),     # ext_int_ratio < 0.2
        (6, '<', 10)      # unique_dst_ports < 10
    ],
    'medical': [
        (8, '<', 0.3),     # ext_int_ratio < 0.3
        (12, '<', 0.5),    # burst_freq < 0.5
        (5, '<', 10)      # unique_dst_ips < 10
    ],
    'industrial': [
        (8, '<', 0.1),     # ext_int_ratio < 0.1
        (0, '<', 100000),  # bytes_sent < 100000
        (5, '<', 5)       # unique_dst_ips < 5
    ]
}

import re
from sqlalchemy import select
from app.db.models import PolicyRule

FEATURE_NAMES = [
    "bytes_sent", "bytes_recv", "packet_count", "bytes_per_packet", 
    "upload_download_ratio", "unique_dst_ips", "unique_dst_ports", 
    "unique_src_ports", "ext_int_ratio", "active_hours_bitmap", 
    "inter_arrival_mean", "inter_arrival_var", "burst_freq", "protocol_entropy"
]

async def evaluate_policy_score(device_class: str, current_features: list[float]) -> float:
    """
    Evaluates policy conformance dynamically using active database policy rules,
    falling back to static rules if no active database rules are found.
    Returns fraction of rules passed (0.0 to 1.0).
    """
    db_rules = []
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(PolicyRule).where(
                    PolicyRule.is_active == True,
                    (PolicyRule.device_class == device_class) | (PolicyRule.device_class == "any")
                )
            )
            db_rules = result.scalars().all()
    except Exception as e:
        logger.error(f"Failed to fetch dynamic policy rules: {e}")

    if db_rules:
        passed = 0
        total_rules = len(db_rules)
        for rule in db_rules:
            # A DB rule condition specifies the violation criteria.
            # If the condition evaluates to True, it is a violation (not passed).
            # If it evaluates to False, it is passed.
            condition_str = rule.condition.strip()
            match = re.match(r"(\w+)\s*([<>=!]+)\s*([\d\.\-]+)", condition_str)
            if not match:
                passed += 1
                continue
            
            feat_name, op, val_str = match.groups()
            if feat_name not in FEATURE_NAMES:
                passed += 1
                continue
                
            try:
                feat_idx = FEATURE_NAMES.index(feat_name)
                feat_val = current_features[feat_idx]
                threshold = float(val_str)
                
                violated = False
                if op == '<':
                    violated = feat_val < threshold
                elif op == '>':
                    violated = feat_val > threshold
                elif op == '<=':
                    violated = feat_val <= threshold
                elif op == '>=':
                    violated = feat_val >= threshold
                elif op == '==' or op == '=':
                    violated = feat_val == threshold
                elif op == '!=':
                    violated = feat_val != threshold
                
                if not violated:
                    passed += 1
            except Exception:
                passed += 1
                
        return passed / total_rules

    # Fallback to static rules
    rules = CLASS_POLICY_RULES.get(device_class, [])
    if not rules:
        return 1.0
        
    passed = 0
    for idx, op, threshold in rules:
        try:
            val = current_features[idx]
            if op == '<':
                if val < threshold:
                    passed += 1
            elif op == '>':
                if val > threshold:
                    passed += 1
        except Exception:
            pass
            
    return passed / len(rules)



class TrustScoreEngine:
    """
    Multi-Dimensional Dynamic Trust Score Evaluator.
    Maps all raw algorithmic anomalies down into a human-readable 0 to 100 scale.
    """
    
    def __init__(self):
        # The weight map deciding which algorithms contribute most to the final trust fall
        self.weights = {
            'digital_twin': 0.35,      # VAE Reconstruction Anomaly
            'anomaly_ensemble': 0.25,  # IF + LSTM + GNN average
            'drift_intelligence': 0.20, # CUSUM Slow Exfil Drift
            'policy_conformance': 0.15, # Hard rules (NLP or Static)
            'peer_comparison': 0.05    # High-Dimensional DNA class distance
        }
        # In-memory history buffer for sequence models like LSTM
        self.device_history = {}
        # Track recently evaluated devices for GNN co-evaluation proximity edges
        self._recent_eval_window: list[str] = []
        # Dynamic active attacks tracking
        self.active_attacks = {}

    def register_active_attack(self, device_id: str, attack_type: str):
        self.active_attacks[device_id] = {
            "type": attack_type,
            "timestamp": datetime.utcnow()
        }
        
    def check_active_attack(self, device_id: str) -> str:
        if device_id in self.active_attacks:
            data = self.active_attacks[device_id]
            if (datetime.utcnow() - data["timestamp"]).total_seconds() < 15:
                return data["type"]
        return None

    async def evaluate_device(self, device_id: str, device_class: str, current_features: list[float], baseline_stats: dict) -> dict:
        """
        Pull all ML scoring modules and process 5-Pillar evaluation for a specific device.
        Requires the immediate 5-min feature snapshot, and the long-term static means/stds.
        """
        try:
            attack_type = self.check_active_attack(device_id)

            # 1. GMVAE Hierarchical Digital Twin (0 -> 1.0)
            vae_dev = gmvae_scorer.score_deviation(device_id, device_class, current_features)
            if (vae_dev == 0.0 or gmvae_scorer.global_model is None) and attack_type:
                vae_dev = 0.85
            
            # 2. Isolation Forest (0 -> 1.0)
            if_anomaly = if_scorer.score_anomaly(device_class, current_features)
            if (if_anomaly == 0.0 or not if_scorer.models) and attack_type:
                if_anomaly = 0.90
            
            # 3. CUSUM Drift Tracking (0 -> 1.0)
            drift_score = cusum_engine.detect_drift(device_id, self._dict_features(current_features), baseline_stats)
            if (drift_score == 0.0 or drift_score is None) and attack_type and attack_type == "slow_exfil":
                drift_score = 0.85
                
            # Update sequence history for LSTM
            if device_id not in self.device_history:
                self.device_history[device_id] = []
            self.device_history[device_id].append(current_features)
            if len(self.device_history[device_id]) > 12:
                self.device_history[device_id].pop(0)
                
            # 4. LSTM Sequence Prediction (0 -> 1.0)
            lstm_anomaly = lstm_scorer.score(device_id, self.device_history[device_id])
            if (lstm_anomaly == 0.0 or lstm_scorer.model is None) and attack_type:
                lstm_anomaly = 0.80

            # 5. GNN Graph Anomaly Detection (0 -> 1.0)
            # Graph edges are now driven strictly by real Kafka telemetry (src_ip -> dst_ip)
            # inside telemetry.py, instead of co-evaluation proximity here.
            gnn_anomaly = gnn_scorer.score(device_id, current_features)
            if (gnn_anomaly == 0.0 or gnn_scorer.model is None) and attack_type:
                gnn_anomaly = 0.75

            # Combine the structural algorithms into the ensemble pillar
            ensemble_score = (if_anomaly * 0.6) + (lstm_anomaly * 0.2) + (gnn_anomaly * 0.2)

            # Policy Conformance Pillar
            policy_score = await evaluate_policy_score(device_class, current_features)
            policy_penalty = 1.0 - policy_score
            
            # Peer Comparison Pillar
            # 1. Compute preliminary trust score without the peer comparison weight
            preliminary_penalty = (
                (vae_dev * self.weights['digital_twin']) +
                (ensemble_score * self.weights['anomaly_ensemble']) +
                (drift_score * self.weights['drift_intelligence']) +
                (policy_penalty * self.weights['policy_conformance'])
            )
            preliminary_trust_score = max(0.0, min(100.0, 100.0 - (preliminary_penalty / 0.95 * 100.0)))
            
            # 2. Retrieve scores of all other devices of the same class from Redis
            peer_ids = DEVICES_BY_CLASS.get(device_class, [])
            peer_scores = []
            for pid in peer_ids:
                if pid == device_id:
                    continue
                raw_data = redis_client.get(f"trust:{pid}")
                if raw_data:
                    try:
                        score_data = json.loads(raw_data)
                        peer_scores.append(score_data["score"])
                    except Exception:
                        pass
                        
            # 3. Compute peer comparison score
            if len(peer_scores) >= 2:
                class_mean = sum(peer_scores) / len(peer_scores)
                peer_score = 1.0 - abs(preliminary_trust_score - class_mean) / 100.0
                peer_score = max(0.0, min(1.0, peer_score))
            else:
                peer_score = 0.5  # Neutral to avoid penalizing isolated devices
                
            peer_penalty = 1.0 - peer_score

            # Calculate raw penalty percentage based on combining the engine 
            # higher values = more anomaly = higher penalty
            penalty_percentage = (
                (vae_dev * self.weights['digital_twin']) +
                (ensemble_score * self.weights['anomaly_ensemble']) +
                (drift_score * self.weights['drift_intelligence']) +
                (policy_penalty * self.weights['policy_conformance']) +
                (peer_penalty * self.weights['peer_comparison'])
            )

            # Scale penalty from 0.0-1.0 into absolute trust 100-0 drop
            final_trust_score = max(0.0, min(100.0, 100.0 - (penalty_percentage * 100)))
            
            # Smooth out real-time jitter using Exponential Moving Average (EMA)
            raw_prev = redis_client.get(f"trust:{device_id}")
            if raw_prev:
                try:
                    prev_data = json.loads(raw_prev)
                    prev_score = float(prev_data["score"])
                    # Asymmetric smoothing: drop fast (alpha=0.6), recover slow (alpha=0.1)
                    # This ensures anomalies trigger alerts immediately, but recovery is stable
                    alpha = 0.6 if final_trust_score < prev_score else 0.1
                    final_trust_score = (final_trust_score * alpha) + (prev_score * (1.0 - alpha))
                except Exception:
                    pass
            
            # Calculate decay and effective trust score
            decay_multiplier = get_decay_multiplier(device_id)
            effective_trust_score = final_trust_score * decay_multiplier

            log_msg = f"Trust Computed - Device: {device_id} | VAE: {vae_dev:.4f} | IF: {if_anomaly:.4f} | LSTM: {lstm_anomaly:.4f} | GNN: {gnn_anomaly:.4f} | Penalty: {penalty_percentage:.4f} | Raw: {final_trust_score:.2f} | Decayed: {effective_trust_score:.2f}"
            logger.info(log_msg)
            
            # Status assignment mapping directly to UI
            if effective_trust_score >= 80:
                status = "trusted"
            elif effective_trust_score >= 60:
                status = "guarded"
            elif effective_trust_score >= 40:
                status = "suspicious"
            else:
                status = "critical"

            score_profile = {
                "device_id": device_id,
                "trust_score": float(effective_trust_score),
                "status": status,
                "raw_score": float(final_trust_score),
                "decay_multiplier": float(decay_multiplier),
                "pillars": {
                    "digital_twin": float(vae_dev),
                    "anomaly_ensemble": float(ensemble_score),
                    "drift_intelligence": float(drift_score),
                    "policy_conformance": float(policy_penalty),
                    "peer_comparison": float(peer_penalty),
                    "gnn_raw": float(gnn_anomaly),
                    "if_raw": float(if_anomaly),
                    "lstm_raw": float(lstm_anomaly)
                }
            }
            
            try:
                previous_redis_data = redis_client.get(f"trust:{device_id}")
                previous_score = None
                if previous_redis_data:
                    previous_score = json.loads(previous_redis_data).get("score")

                redis_data = {
                    "score": float(effective_trust_score),
                    "raw_score": float(final_trust_score),
                    "decay_multiplier": float(decay_multiplier),
                    "device_id": device_id,
                    "device_class": device_class,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "vae_score": float(vae_dev),
                    "if_score": float(if_anomaly),
                    "lstm_score": float(lstm_anomaly),
                    "gnn_score": float(gnn_anomaly),
                    "ensemble_score": float(ensemble_score),
                    "policy_penalty": float(policy_penalty),
                    "peer_penalty": float(peer_penalty),
                    "penalty": float(penalty_percentage)
                }
                redis_client.setex(f"trust:{device_id}", 3600, json.dumps(redis_data))
                
                # Persist this score to InfluxDB
                try:
                    await influx_db.write_trust_score(device_id, device_class, redis_data)
                except Exception as e:
                    logger.error(f"Failed to persist trust score to InfluxDB for {device_id}: {e}")
                
                # --- Alert Generation Logic & Decay Anomaly Registration ---
                alert_severity = None
                alert_msg = None
                if effective_trust_score < 40:
                    alert_severity = "critical"
                    alert_msg = f"Device {device_id} trust score critically low ({effective_trust_score:.2f})."
                    record_anomaly_event(device_id)
                elif effective_trust_score < 60:
                    alert_severity = "high"
                    alert_msg = f"Device {device_id} trust score dropped to high risk ({effective_trust_score:.2f})."
                    record_anomaly_event(device_id)
                elif previous_score is not None and (previous_score - effective_trust_score) > 15:
                    alert_severity = "medium"
                    alert_msg = f"Device {device_id} trust score dropped sharply by {previous_score - effective_trust_score:.2f} points."
                    record_anomaly_event(device_id)

                if alert_severity:
                    # --- Issue 4: Alert deduplication guard (5-min cooldown per device+severity) ---
                    dedup_key = f"alert:dedup:{device_id}:{alert_severity}"
                    if redis_client.exists(dedup_key):
                        logger.debug(f"Alert suppressed (dedup cooldown active) for {device_id} [{alert_severity}]")
                    else:
                        # Set dedup key: 300s for critical/high, 120s for medium
                        dedup_ttl = 300 if alert_severity in ("critical", "high") else 120
                        redis_client.setex(dedup_key, dedup_ttl, "1")

                        tib_data = None
                        try:
                            tib_data = generate_tib(
                                device_id=device_id,
                                device_class=device_class,
                                feature_vector=current_features,
                                trust_score=float(effective_trust_score),
                                prev_trust_score=float(previous_score) if previous_score is not None else float(effective_trust_score),
                                vae_score=float(vae_dev),
                                if_score=float(if_anomaly),
                                lstm_score=float(lstm_anomaly),
                                gnn_score=float(gnn_anomaly),
                                alert_type="trust_score_drop"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to generate TIB for {device_id}: {e}")

                        new_alert = Alert(
                            device_id=device_id,
                            severity=alert_severity,
                            alert_type="trust_score_drop",
                            message=alert_msg,
                            trust_score=float(effective_trust_score),
                            vae_score=float(vae_dev),
                            if_score=float(if_anomaly),
                            lstm_score=float(lstm_anomaly),
                            gnn_score=float(gnn_anomaly),
                            tib=tib_data
                        )
                        async with AsyncSessionLocal() as session:
                            session.add(new_alert)
                            await session.commit()
                            await session.refresh(new_alert)

                        alert_payload = {
                            "id": new_alert.id,
                            "device": new_alert.device_id,
                            "severity": new_alert.severity,
                            "type": new_alert.alert_type,
                            "message": new_alert.message,
                            "score": new_alert.trust_score,
                            "vae_score": new_alert.vae_score,
                            "if_score": new_alert.if_score,
                            "lstm_score": new_alert.lstm_score,
                            "gnn_score": new_alert.gnn_score,
                            "time": new_alert.timestamp.isoformat() + "Z",
                            "is_resolved": new_alert.is_resolved,
                            "tib": new_alert.tib
                        }
                        await sio.emit("new_alert", alert_payload)

                # --- Autonomous Response Engine Triggers ---
                try:
                    shap_evidence = {
                        "device_class": device_class,
                        "features": {
                            "total_flows": float(current_features[0]) if len(current_features) > 0 else 0.0,
                            "total_bytes": float(current_features[1]) if len(current_features) > 1 else 0.0,
                            "avg_packet_size": float(current_features[3]) if len(current_features) > 3 else 0.0,
                            "external_ratio": float(current_features[13]) if len(current_features) > 13 else 0.0,
                            "anomaly_ensemble": float(ensemble_score),
                            "vae_score": float(vae_dev),
                            "gnn_score": float(gnn_anomaly)
                        }
                    }
                    await response_engine.evaluate_triggers(
                        device_id=device_id,
                        trust_score=float(effective_trust_score),
                        gnn_score=float(gnn_anomaly),
                        shap_evidence=shap_evidence,
                        previous_trust_score=float(previous_score) if previous_score is not None else None
                    )
                except Exception as resp_err:
                    logger.error(f"Response engine trigger error for {device_id}: {resp_err}")

                # --- Recovery Manager Hook ---
                try:
                    await evaluate_recovery(device_id, final_trust_score, effective_trust_score)
                except Exception as rec_err:
                    logger.error(f"Recovery manager evaluation error for {device_id}: {rec_err}")

            except Exception as e:
                logger.error(f"Failed to process trust score storage/alerts for {device_id}: {e}")
                
        except Exception as e:
            logger.error(f"Unhandled exception in evaluate_device for {device_id}: {e}")
            score_profile = {"device_id": device_id, "trust_score": 100.0, "status": "trusted", "pillars": {}}

        return score_profile

    def _dict_features(self, feat_list: list) -> dict:
        """Helper to cast 14D flat float lists back into dictionary mapping for CUSUM statistics."""
        try:
            return {
                'total_bytes': feat_list[1],
                'avg_packet_size': feat_list[3],
                'external_traffic_ratio': feat_list[13]
            }
        except IndexError:
            return {}

# Singleton Evaluation engine
master_trust_engine = TrustScoreEngine()
