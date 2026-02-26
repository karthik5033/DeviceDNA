import logging
from app.ml.vae.scoring import twin_scorer
from app.ml.isolation_forest.model import if_scorer
from app.services.drift_engine import cusum_engine
from app.db.influxdb import influx_db

logger = logging.getLogger(__name__)

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

    async def evaluate_device(self, device_id: str, device_class: str, current_features: list[float], baseline_stats: dict) -> dict:
        """
        Pull all ML scoring modules and process 5-Pillar evaluation for a specific device.
        Requires the immediate 5-min feature snapshot, and the long-term static means/stds.
        """
        try: # 1. VAE Digital Twin (0 -> 1.0)
            vae_dev = twin_scorer.score_deviation(device_id, current_features)
        except Exception:
            vae_dev = 0.0
            
        try: # 2. Isolation Forest (0 -> 1.0)
            if_anomaly = if_scorer.score_anomaly(device_class, current_features)
        except Exception:
            if_anomaly = 0.0
            
        try: # 3. CUSUM Drift Tracking (0 -> 1.0)
            drift_score = cusum_engine.detect_drift(device_id, self._dict_features(current_features), baseline_stats)
        except Exception:
            drift_score = 0.0
            
        # Combine the structural algorithms into the ensemble pillar
        # Assuming LSTM and GNN (GraphSAGE) evaluate as 0 currently pending GPU implementations
        ensemble_score = (if_anomaly * 0.6) + (0.0 * 0.2) + (0.0 * 0.2)

        # Assuming Policy violations = 0 for default flow
        policy_penalty = 0.0
        
        # DNA Cross-Validation (Assuming exact match = 0 Penalty)
        peer_penalty = 0.0

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
        
        # Status assignment mapping directly to UI
        if final_trust_score >= 80:
            status = "trusted"
        elif final_trust_score >= 60:
            status = "guarded"
        elif final_trust_score >= 40:
            status = "suspicious"
        else:
            status = "critical"

        score_profile = {
            "device_id": device_id,
            "trust_score": float(final_trust_score),
            "status": status,
            "pillars": {
                "digital_twin": float(vae_dev),
                "anomaly_ensemble": float(ensemble_score),
                "drift_intelligence": float(drift_score),
                "policy_conformance": float(policy_penalty),
                "peer_comparison": float(peer_penalty)
            }
        }
        
        # NOTE: Missing feature - Persisting this score to Redis and InfluxDB
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
