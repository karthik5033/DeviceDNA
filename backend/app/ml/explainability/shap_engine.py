import numpy as np
import shap
import logging
from app.ml.vae.scoring import twin_scorer
from app.ml.isolation_forest.model import if_scorer

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "bytes_sent", "bytes_recv", "packet_count", "bytes_per_packet", 
    "upload_download_ratio", "unique_dst_ips", "unique_dst_ports", 
    "unique_src_ports", "ext_int_ratio", "active_hours_bitmap", 
    "inter_arrival_mean", "inter_arrival_var", "burst_freq", "protocol_entropy"
]

def explain_anomalies(device_id: str, device_class: str, feature_vector: list[float]) -> dict:
    """
    Accepts a device's current 14-dim feature vector and its device class.
    Runs KernelSHAP on the VAE reconstruction error and TreeSHAP on the Isolation Forest.
    Returns the top 5 contributing features by absolute SHAP value.
    """
    if len(feature_vector) != 14:
        raise ValueError("Feature vector must be exactly 14-dimensional.")
        
    X = np.array(feature_vector).reshape(1, -1)
    
    # -----------------------------------------
    # 1. TreeSHAP for Isolation Forest
    # -----------------------------------------
    if_top_features = []
    if device_class in if_scorer.models:
        model_if = if_scorer.models[device_class]
        
        # Isolation Forest expected value (baseline)
        explainer_if = shap.TreeExplainer(model_if)
        
        # For sklearn IsolationForest, shap_values might be a list or array
        shap_values_if = explainer_if.shap_values(X)
        
        if isinstance(shap_values_if, list):
            shap_values_if = shap_values_if[0]
            
        expected_if = explainer_if.expected_value
        if isinstance(expected_if, (list, np.ndarray)):
            expected_if = expected_if[0]
            
        sv_if = shap_values_if[0]  # shape: (14,)
        
        # Get top 5 by absolute magnitude
        top_indices_if = np.argsort(np.abs(sv_if))[::-1][:5]
        
        for idx in top_indices_if:
            val = float(sv_if[idx])
            if_top_features.append({
                "feature_name": FEATURE_NAMES[idx],
                "shap_value": val,
                "observed_value": float(X[0, idx]),
                "baseline_value": float(expected_if),
                "direction": "increase" if val > 0 else "decrease"
            })
            
    # -----------------------------------------
    # 2. KernelSHAP for VAE
    # -----------------------------------------
    vae_top_features = []
    
    if device_id in twin_scorer.twins:
        # Wrapper to get VAE anomaly scores
        def vae_predict(X_batch):
            scores = []
            for row in X_batch:
                scores.append(twin_scorer.score(device_id, row.tolist()))
            return np.array(scores)
            
        # Use device minimums as baseline for KernelSHAP
        norm_params = twin_scorer.twins[device_id]['norm']
        f_mins = norm_params.get('min', norm_params.get('mins', norm_params.get('feature_mins', [0]*14)))
        background = np.array(f_mins).reshape(1, -1)
        
        # Kernel explainer
        explainer_vae = shap.KernelExplainer(vae_predict, background)
        
        # Calculate shap values - disable tqdm printout for cleaner logs
        shap_values_vae = explainer_vae.shap_values(X, nsamples=100, silent=True) 
        
        if isinstance(shap_values_vae, list):
            shap_values_vae = shap_values_vae[0]
            
        expected_vae = explainer_vae.expected_value
        if isinstance(expected_vae, (list, np.ndarray)):
            expected_vae = expected_vae[0]
            
        sv_vae = shap_values_vae[0]
        
        top_indices_vae = np.argsort(np.abs(sv_vae))[::-1][:5]
        
        for idx in top_indices_vae:
            val = float(sv_vae[idx])
            vae_top_features.append({
                "feature_name": FEATURE_NAMES[idx],
                "shap_value": val,
                "observed_value": float(X[0, idx]),
                "baseline_value": float(expected_vae),
                "direction": "increase" if val > 0 else "decrease"
            })
            
    return {
        "isolation_forest": if_top_features,
        "vae": vae_top_features
    }

if __name__ == "__main__":
    test_device = "SIM-0001"
    test_class = "camera"
    test_features = [1200.0, 4500.0, 10.0, 120.0, 0.2, 2.0, 2.0, 1.0, 0.5, 255.0, 10.0, 1.0, 0.1, 0.5]
    
    print(f"Testing SHAP engine for {test_device} ({test_class})")
    try:
        results = explain_anomalies(test_device, test_class, test_features)
        print("\n=== ISOLATION FOREST TOP 5 ===")
        for r in results.get("isolation_forest", []):
            print(f"- {r['feature_name']}: shap={r['shap_value']:.4f} (observed={r['observed_value']}, {r['direction']})")
            
        print("\n=== VAE TOP 5 ===")
        for r in results.get("vae", []):
            print(f"- {r['feature_name']}: shap={r['shap_value']:.4f} (observed={r['observed_value']}, {r['direction']})")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
