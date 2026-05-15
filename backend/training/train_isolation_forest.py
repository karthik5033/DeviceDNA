import os
import json
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the exact 14-dimensional feature distributions (means, stds) for the 6 device classes
# 14 features: 
# 0: total_flows, 1: total_bytes, 2: total_packets, 3: avg_packet_size, 4: avg_duration_ms
# 5: tcp_ratio, 6: udp_ratio, 7: http_ratio, 8: https_ratio, 9: dns_ratio, 10: other_protocol_ratio
# 11: unique_dst_ips, 12: unique_dst_ports, 13: external_traffic_ratio
PROFILES = {
    'camera': {
        'means': [80.0, 1200000.0, 1520.0, 782.0, 2500.0, 0.85, 0.15, 0.25, 0.15, 0.10, 0.50, 3.0, 4.0, 0.15],
        'stds':  [15.0, 300000.0,  300.0,  100.0, 500.0,  0.05, 0.05, 0.05, 0.05, 0.02, 0.05, 1.0, 1.0, 0.05]
    },
    'sensor': {
        'means': [20.0, 10000.0, 60.0, 160.0, 2500.0, 0.80, 0.20, 0.20, 0.0, 0.15, 0.65, 2.0, 2.0, 0.01],
        'stds':  [ 5.0,  2000.0, 10.0,  20.0,  500.0, 0.05, 0.05, 0.05, 0.0, 0.05, 0.05, 0.0, 0.0, 0.01]
    },
    'thermostat': {
        'means': [15.0, 15000.0, 50.0, 288.0, 2500.0, 0.60, 0.40, 0.0, 0.50, 0.25, 0.25, 3.0, 3.0, 0.30],
        'stds':  [ 5.0,  5000.0, 15.0,  40.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 0.10]
    },
    'access_control': {
        'means': [30.0, 60000.0, 110.0, 544.0, 2500.0, 0.70, 0.30, 0.0, 0.40, 0.20, 0.40, 4.0, 3.0, 0.10],
        'stds':  [10.0, 20000.0,  30.0,  50.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 0.05]
    },
    'medical': {
        'means': [40.0, 320000.0, 150.0, 2080.0, 2500.0, 0.80, 0.20, 0.0, 0.30, 0.10, 0.60, 5.0, 4.0, 0.05],
        'stds':  [12.0, 100000.0,  40.0,  200.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 2.0, 1.0, 0.03]
    },
    'industrial': {
        'means': [50.0, 150000.0, 140.0, 1056.0, 2500.0, 0.80, 0.20, 0.0, 0.0, 0.10, 0.90, 3.0, 3.0, 0.02],
        'stds':  [15.0,  40000.0,  35.0,  100.0,  500.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 1.0, 0.0, 0.01]
    }
}

def train_models():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models_trained")
    os.makedirs(models_dir, exist_ok=True)
    
    n_samples = 500
    
    for cls_name, params in PROFILES.items():
        means = np.array(params['means'])
        stds = np.array(params['stds'])
        
        # Generate 500 normal samples using numpy random gaussian sampling
        np.random.seed(42)
        X = np.random.normal(loc=means, scale=stds, size=(n_samples, len(means)))
        
        # Ensure ratios and non-negative values are somewhat bounded
        X = np.maximum(X, 0.0) 
        
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        model.fit(X)
        
        model_path = os.path.join(models_dir, f"if_{cls_name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Successfully trained and saved Isolation Forest for '{cls_name}' to {model_path}")

if __name__ == "__main__":
    train_models()
