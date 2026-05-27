import logging
from app.ml.explainability.shap_engine import explain_anomalies
from app.ml.explainability.feature_language import feature_to_statement

logger = logging.getLogger(__name__)

# Static lookup for recommended actions by severity
_RECOMMENDED_ACTIONS = {
    "Critical": [
        "Isolate device from network segment immediately",
        "Preserve forensic state — enable full packet capture",
        "Conduct threat hunt for lateral movement to adjacent devices"
    ],
    "High": [
        "Quarantine device to restricted VLAN",
        "Review recent authentication logs for the device",
        "Verify firmware integrity and patch level"
    ],
    "Suspicious": [
        "Increase monitoring telemetry frequency",
        "Verify device owner/location in asset inventory",
        "Check for recent legitimate configuration changes"
    ],
    "Guarded": [
        "Monitor for further degradation in trust score",
        "Review baseline threshold appropriateness",
        "No immediate action required"
    ]
}

def generate_tib(
    device_id: str, 
    device_class: str, 
    feature_vector: list[float], 
    trust_score: float, 
    prev_trust_score: float, 
    vae_score: float, 
    if_score: float, 
    lstm_score: float, 
    gnn_score: float, 
    alert_type: str
) -> dict:
    """
    Generates a Threat Intelligence Brief (TIB) for a device.
    Uses SHAP explainability and static mappings to provide human-readable context.
    """
    
    # 1. Determine Severity
    if trust_score < 20:
        severity = "Critical"
    elif trust_score < 40:
        severity = "High"
    elif trust_score < 65:
        severity = "Suspicious"
    else:
        severity = "Guarded"
        
    # 2. Extract SHAP Explanations
    try:
        shap_results = explain_anomalies(device_id, device_class, feature_vector)
        vae_top_features = shap_results.get("vae", [])
        
        evidence_list = []
        top_feature_name = "unknown behavior"
        
        for idx, feature_dict in enumerate(vae_top_features):
            stmt = feature_to_statement(feature_dict)
            evidence_list.append(stmt)
            if idx == 0:
                top_feature_name = feature_dict.get("feature_name", "unknown behavior")
                
        # 3. Construct Headline
        # Since we must keep it pure Python (no LLM text generation), we assemble a clear sentence.
        if evidence_list:
            headline = f"{device_class.capitalize()} {device_id} triggered a '{alert_type}' alert. Primary indicator: {evidence_list[0]}"
        else:
            headline = f"{device_class.capitalize()} {device_id} triggered a '{alert_type}' alert. No specific feature evidence available."
    except Exception as e:
        logger.error(f"SHAP explanation failed for {device_id}: {e}")
        evidence_list = []
        headline = "Anomalous behavior detected"
        
    # 4. Calculate Scores
    trust_score_delta = round(trust_score - prev_trust_score, 2)
    confidence_score = round((vae_score + if_score) / 2.0, 2)
    
    model_scores = {
        "vae": round(vae_score, 3),
        "if_score": round(if_score, 3),
        "lstm": round(lstm_score, 3),
        "gnn": round(gnn_score, 3)
    }
    
    # 5. Build TIB
    tib = {
        "headline": headline,
        "trust_score_current": round(trust_score, 2),
        "trust_score_previous": round(prev_trust_score, 2),
        "trust_score_delta": trust_score_delta,
        "confidence_score": confidence_score,
        "severity": severity,
        "evidence_list": evidence_list,
        "model_scores": model_scores,
        "recommended_actions": _RECOMMENDED_ACTIONS.get(severity, _RECOMMENDED_ACTIONS["Guarded"])
    }
    
    return tib

if __name__ == "__main__":
    import json
    # Dummy data for SIM-0014 (Camera)
    test_device = "SIM-0014"
    test_class = "camera"
    test_features = [120000.0, 500.0, 150.0, 800.0, 240.0, 5.0, 1.0, 15.0, 10.0, 255.0, 2.0, 1.0, 0.9, 0.2]
    
    tib_result = generate_tib(
        device_id=test_device,
        device_class=test_class,
        feature_vector=test_features,
        trust_score=15.4,
        prev_trust_score=78.2,
        vae_score=0.95,
        if_score=0.88,
        lstm_score=0.92,
        gnn_score=0.85,
        alert_type="Data Exfiltration"
    )
    
    print("=" * 80)
    print(f"THREAT INTELLIGENCE BRIEF (TIB) - {test_device}")
    print("=" * 80)
    print(json.dumps(tib_result, indent=2))
