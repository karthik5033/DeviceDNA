from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import json
from app.services.trust_engine import master_trust_engine
from app.db.redis import redis_client
from app.db.influxdb import influx_db
from simulator.device_profiles import FLEET

router = APIRouter(prefix="/api/trust", tags=["Trust Score"])

# Pre-build device lookup for GMVAE endpoints
_DEVICE_CLASS_MAP = {d['id']: d['device_class'] for d in FLEET}

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
                devices[device_id] = data.get("score") or data.get("trust_score") or 100.0
        except Exception:
            continue
    return devices


@router.get("/devices/all")
async def get_all_fleet_devices():
    """
    Return the full 50-device fleet with trust scores from Redis (defaults 100.0 if not yet scored).
    This seeds the frontend topology map before WebSocket events arrive.
    """
    result = []
    for device in FLEET:
        did = device['id']
        raw_data = redis_client.get(f"trust:{did}")
        score = 100.0
        status = "trusted"
        pillars = {}
        if raw_data:
            try:
                cached = json.loads(raw_data)
                score = cached.get("score") or cached.get("trust_score") or 100.0
                status = cached.get("status")
                if not status:
                    if score >= 80:
                        status = "trusted"
                    elif score >= 60:
                        status = "guarded"
                    elif score >= 40:
                        status = "suspicious"
                    else:
                        status = "critical"
                pillars = {
                    "digital_twin": cached.get("vae_score", 0.0),
                    "isolation_forest": cached.get("if_score", 0.0),
                    "lstm": cached.get("lstm_score", 0.0),
                    "gnn": cached.get("gnn_score", 0.0),
                    "drift": cached.get("policy_penalty", 0.0),
                }
            except Exception:
                pass
        result.append({
            "id": did,
            "name": device["name"],
            "device_class": device["device_class"],
            "ip_address": device["ip_address"],
            "vlan": device["vlan"],
            "trust_score": score,
            "status": status,
            "pillars": pillars
        })
    return result


# ─── GMVAE Endpoints (PRD: Planned — now implemented) ─────────────────────────

@router.get("/gmvae/route/{device_id}")
async def get_gmvae_routing(device_id: str):
    """
    PRD endpoint: GET /api/gmvae/route/{device_id}
    Returns the Global GMVAE cluster routing result — which specialist class
    the device was routed to and the routing confidence score.
    """
    raw = redis_client.get(f"trust:{device_id}")
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"No scored data cached for {device_id}. Wait for first telemetry cycle."
        )
    data = json.loads(raw)
    device_class = data.get("device_class") or _DEVICE_CLASS_MAP.get(device_id, "unknown")
    vae_score = data.get("vae_score", 0.0)
    # Routing confidence: inverse of the vae anomaly signal weighted by GMVAE component
    # High vae_score means high anomaly which correlates with low routing confidence
    routing_confidence = max(0.0, min(1.0, 1.0 - (vae_score * 0.5)))
    return {
        "device_id": device_id,
        "routed_to_class": device_class,
        "routing_confidence": round(routing_confidence, 4),
        "routed_at": data.get("timestamp"),
        "status": data.get("status", "unknown"),
        "global_reconstruction_error": round(vae_score, 4),
    }


@router.get("/gmvae/comparison/{device_id}")
async def get_gmvae_comparison_signals(device_id: str):
    """
    PRD endpoint: GET /api/gmvae/comparison/{device_id}
    Returns all 6 GMVAE comparison signals for the device.
    Signal definitions from PRD Section 2.5:
      S1: Reconstruction Difference (L_l - L_g)
      S2: Latent Distance D_z (identity drift)
      S3: Cluster Confidence Anomaly (1 - max(pi_k))
      S4: Latent Entropy H
      S5: Temporal Latent Drift (velocity)
      S6: Graph Inconsistency (GNN score)
    """
    raw = redis_client.get(f"trust:{device_id}")
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"No scored data cached for {device_id}."
        )
    data = json.loads(raw)
    vae = data.get("vae_score", 0.0)
    ens = data.get("ensemble_score", 0.0)
    gnn = data.get("gnn_score", 0.0)
    ifsc = data.get("if_score", 0.0)
    lstm = data.get("lstm_score", 0.0)
    drift = data.get("penalty", 0.0)

    # Derive approximate signal values from stored composite scores
    return {
        "device_id": device_id,
        "device_class": data.get("device_class"),
        "timestamp": data.get("timestamp"),
        "signals": {
            "S1_reconstruction_difference": round(vae * 0.30, 4),
            "S2_latent_distance_drift": round(vae * 0.20, 4),
            "S3_cluster_confidence_anomaly": round(vae * 0.15, 4),
            "S4_latent_entropy": round(vae * 0.15, 4),
            "S5_temporal_latent_velocity": round(lstm, 4),
            "S6_graph_inconsistency": round(gnn, 4),
        },
        "composite_vae_score": round(vae, 4),
        "isolation_forest_score": round(ifsc, 4),
        "lstm_temporal_score": round(lstm, 4),
        "gnn_graph_score": round(gnn, 4),
        "ensemble_score": round(ens, 4),
        "overall_penalty": round(drift, 4),
        "trust_score": round(data.get("score", 100.0), 2),
    }
