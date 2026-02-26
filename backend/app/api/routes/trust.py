from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.services.trust_engine import master_trust_engine

router = APIRouter(prefix="/api/trust", tags=["Trust Score"])

# Mock state tracking (Real would be Redis/PostgreSQL fetched)
# Because currently there's no continuous aggregator saving to DB, we simulate live eval points
MOCK_BASELINE_STATS = {
    'total_bytes': {'mean': 500.0, 'std': 100.0},
    'avg_packet_size': {'mean': 128.0, 'std': 10.0},
    'external_traffic_ratio': {'mean': 0.05, 'std': 0.01}
}

class EvaluateRequest(BaseModel):
    device_id: str
    device_class: str
    current_features: list[float]

@router.post("/evaluate")
async def evaluate_device_trust(payload: EvaluateRequest):
    """
    Force an immediate live computation of the 100-point trust score using the 
    5 overarching ML Pillars (VAE, IF, LSTM, GNN, CUSUM drift).
    """
    if len(payload.current_features) != 14:
        raise HTTPException(status_code=400, detail="current_features must contain exactly 14 float dimensions")
        
    evaluation = await master_trust_engine.evaluate_device(
        device_id=payload.device_id,
        device_class=payload.device_class,
        current_features=payload.current_features,
        baseline_stats=MOCK_BASELINE_STATS
    )
    
    # Store history / Push WebSocket to Frontend here
    
    return evaluation
    
@router.get("/{device_id}/current")
async def get_current_trust_score(device_id: str):
    """
    Get the most recently recorded Trust Score metric for a single LAN device.
    Ideal for rendering dashboard details. 
    (Simulated response pending Redis Cache hookup)
    """
    # Assuming the device currently evaluates perfectly without incoming data payload
    return {
        "device_id": device_id,
        "trust_score": 100.0,
        "status": "trusted",
        "pillars": {
            "digital_twin": 0.0,
            "anomaly_ensemble": 0.0,
            "drift_intelligence": 0.0,
            "policy_conformance": 0.0,
            "peer_comparison": 0.0
        }
    }
