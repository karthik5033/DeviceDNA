"""
Response status API routes.

GET /api/response/{device_id}/status
Returns the current active autonomous-response flags for a device,
read directly from Redis — no database involved.
"""

from fastapi import APIRouter
from app.services.response_engine import ResponseEngine, response_engine

router = APIRouter(prefix="/api/response", tags=["Response Engine"])


@router.get("/{device_id}/status")
async def get_response_status(device_id: str):
    """
    Return current active response flags for a device:
    {
        "device_id": "SIM-0001",
        "isolated": false,
        "sandboxed": true,
        "forensic_capture": false,
        "blocked_ips": ["10.0.0.5"]
    }
    """
    return ResponseEngine.get_device_response_status(device_id)

@router.post("/{device_id}/isolate")
async def manual_isolate(device_id: str):
    """
    Manually trigger isolation for a device.
    """
    action_taken = await response_engine.isolate_device(device_id)
    return {"device_id": device_id, "isolated": True, "newly_triggered": action_taken}

