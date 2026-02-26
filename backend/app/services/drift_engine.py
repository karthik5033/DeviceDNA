import logging

logger = logging.getLogger(__name__)

class CUSUMDriftEngine:
    """
    Cumulative Sum (CUSUM) algorithm for detecting slow, sustained deviations
    in behavior that structural ML algorithms (IF, VAE) miss.
    Specifically targets Scenario 2: Slow Data Exfiltration.
    """
    
    def __init__(self, drift_threshold: float = 3.0, slack: float = 0.5):
        # Param K: Allowed variation range (Slack)
        self.k = slack 
        # Param H: Threshold before CUSUM signals an alarm
        self.h = drift_threshold
        
        # State tracking: device_id -> {feature_name -> (sum_pos, sum_neg)}
        self.state_cache = {}

    def init_device_state(self, device_id: str):
        self.state_cache[device_id] = {
            'total_bytes': {'s_pos': 0.0, 's_neg': 0.0},
            'avg_packet_size': {'s_pos': 0.0, 's_neg': 0.0},
            'external_traffic_ratio': {'s_pos': 0.0, 's_neg': 0.0}
        }

    def detect_drift(self, device_id: str, new_features: dict, baseline_stats: dict) -> float:
        """
        Takes the new features and baseline standardized means for critical features
        capable of slow drift. Unpacks mathematical accumulation tracking.
        Returns 0.0 to 1.0 (1.0 = highly mathematically drifted).
        """
        if device_id not in self.state_cache:
            self.init_device_state(device_id)

        drift_score = 0.0
        
        # Target specific slow-exfil features from the PRD
        target_keys = ['total_bytes', 'avg_packet_size', 'external_traffic_ratio']
        
        for key in target_keys:
            if key not in new_features or key not in baseline_stats:
                continue
                
            val_t = new_features[key]
            mean = baseline_stats[key]['mean']
            std = max(baseline_stats[key]['std'], 0.001) # Avoid div 0
            
            # Z-score normalization of this specific timestamp reading
            z_t = (val_t - mean) / std
            
            # CUSUM positive and negative deviations updating
            s_pos = max(0, self.state_cache[device_id][key]['s_pos'] + z_t - self.k)
            s_neg = max(0, self.state_cache[device_id][key]['s_neg'] - z_t - self.k)
            
            # Clip state to the threshold limit + 10 to avoid integer explode overflow
            self.state_cache[device_id][key]['s_pos'] = min(s_pos, self.h * 1.5)
            self.state_cache[device_id][key]['s_neg'] = min(s_neg, self.h * 1.5)
            
            # Check for alarm: Did it cross threshold `H` ?
            feature_drift = max(s_pos, s_neg) / self.h
            
            # We scale the overall score up to 1.0 based on how far past threshold it got
            drift_score = max(drift_score, feature_drift)

        # Normalize total fractional 0 to 1
        return float(max(0.0, min(1.0, drift_score)))

# Singleton Engine
cusum_engine = CUSUMDriftEngine()
