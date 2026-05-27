from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import json
from app.services.trust_engine import master_trust_engine
from app.db.redis import redis_client
from app.db.influxdb import influx_db

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
    """
    raw_data = redis_client.get(f"trust:{device_id}")
    if not raw_data:
        raise HTTPException(status_code=404, detail=f"No recent trust score found for {device_id}")
        
    return json.loads(raw_data)

@router.get("/{device_id}/history")
async def get_trust_history(device_id: str, hours: int = Query(24, description="Number of hours of history to fetch")):
    """
    Get the historical trust scores for a specific device from InfluxDB.
    Returns a list of timestamps and scores. If no history exists, returns [].
    """
    try:
        history = await influx_db.query_trust_history(device_id, hours=hours)
        return history
    except Exception:
        return []

@router.get("/devices")
async def get_all_devices():
    """
    Get all tracked devices and their current trust scores from Redis.
    """
    keys = redis_client.keys("trust:*")
    devices = {}
    for key in keys:
        try:
            # handle bytes if necessary
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            device_id = key_str.split("trust:")[1]
            raw_data = redis_client.get(key)
            if raw_data:
                data = json.loads(raw_data)
                devices[device_id] = data.get("trust_score", 100.0)
        except Exception:
            continue
    return devices
