from fastapi import APIRouter, HTTPException
import json
from app.services.response_engine import response_engine, ResponseEngine
from app.db.redis import redis_client

router = APIRouter(prefix="/api/response", tags=["Response Engine"])

@router.get("/pending")
async def get_pending_responses():
    """
    Get all pending high-risk response actions currently queued in Redis
    waiting for Human-in-the-Loop (HITL) approval.
    """
    cursor = "0"
    pending_actions = []
    try:
        while True:
            cursor, keys = redis_client.scan(cursor=cursor, match="response:pending:*", count=100)
            for k in keys:
                val = redis_client.get(k)
                if val:
                    pending_actions.append(json.loads(val))
            if cursor == 0 or cursor == "0":
                break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan pending Redis actions: {e}")
    return pending_actions

@router.get("/{device_id}/status")
async def get_response_status(device_id: str):
    """
    Return current active response flags and pending status for a device.
    """
    return ResponseEngine.get_device_response_status(device_id)

@router.post("/approve/{device_id}")
async def approve_response(device_id: str):
    """
    Approve a pending high-risk action (Quarantine/Honeypot) for execution.
    """
    pending_key = f"response:pending:{device_id}"
    pending_raw = redis_client.get(pending_key)
    if not pending_raw:
        raise HTTPException(status_code=404, detail="No pending response action found for this device")

    pending = json.loads(pending_raw)
    action = pending.get("action")
    score = pending.get("trigger_score", 30.0)

    # Clear pending state
    redis_client.delete(pending_key)
    
    # Execute actual response action with approved decision logging
    if action == "quarantine":
        await response_engine.isolate_device(device_id, score, hitl_decision="approved")
    elif action == "honeypot":
        await response_engine.honeypot_device(device_id, score, hitl_decision="approved")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported pending action type: {action}")

    return {"status": "success", "device_id": device_id, "action": action, "decision": "approved"}

@router.post("/deny/{device_id}")
async def deny_response(device_id: str):
    """
    Deny a pending high-risk action, preventing execution and setting a 5-minute override.
    """
    pending_key = f"response:pending:{device_id}"
    pending_raw = redis_client.get(pending_key)
    if not pending_raw:
        raise HTTPException(status_code=404, detail="No pending response action found for this device")

    pending = json.loads(pending_raw)
    action = pending.get("action")
    score = pending.get("trigger_score", 30.0)
    tier = pending.get("target_tier", 4)

    # Clear pending state and set a 5-minute ignore key (300 seconds)
    redis_client.delete(pending_key)
    override_key = f"response:override:{device_id}"
    redis_client.setex(override_key, 300, "true")

    # Log denial to PostgreSQL ResponseAuditLog
    await response_engine._log_action_to_db(device_id, score, tier, action, "denied")

    return {"status": "success", "device_id": device_id, "action": action, "decision": "denied"}

@router.post("/{device_id}/isolate")
async def manual_isolate(device_id: str):
    """
    Manually trigger immediate isolation for a device.
    """
    action_taken = await response_engine.isolate_device(device_id, 0.0, hitl_decision="manual_override")
    return {"device_id": device_id, "isolated": True, "newly_triggered": action_taken}
