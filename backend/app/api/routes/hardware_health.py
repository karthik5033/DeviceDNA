from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timezone
import json
import logging
import redis.asyncio as aioredis
from app.db.redis import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

router = APIRouter()

# Async Redis client specifically for the device registry router
redis_client = aioredis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

# Mock state for presentation
PRESENTATION_NODES = {
    "gyro": {"device_id": "gyro", "device_class": "sensor", "source": "physical", "last_seen": datetime.now(timezone.utc).isoformat(), "status": "online"},
    "mq3": {"device_id": "mq3", "device_class": "sensor", "source": "physical", "last_seen": datetime.now(timezone.utc).isoformat(), "status": "online"},
    "mq135": {"device_id": "mq135", "device_class": "sensor", "source": "physical", "last_seen": datetime.now(timezone.utc).isoformat(), "status": "online"},
    "mq2": {"device_id": "mq2", "device_class": "sensor", "source": "physical", "last_seen": datetime.now(timezone.utc).isoformat(), "status": "online"},
    "ldr sensor": {"device_id": "ldr sensor", "device_class": "sensor", "source": "physical", "last_seen": datetime.now(timezone.utc).isoformat(), "status": "online"},
}

@router.get("/devices")
async def get_all_devices() -> List[Dict[str, Any]]:
    """Returns the mock presentation devices."""
    # Update last_seen for online devices to keep them 'Just now'
    for d in PRESENTATION_NODES.values():
        if d["status"] == "online":
            d["last_seen"] = datetime.now(timezone.utc).isoformat()
    return list(PRESENTATION_NODES.values())

@router.post("/devices/{device_id}/toggle")
async def toggle_device_status(device_id: str):
    """Manually toggle a device online/offline for presentation."""
    if device_id not in PRESENTATION_NODES:
        raise HTTPException(status_code=404, detail="Device not found")
        
    current = PRESENTATION_NODES[device_id]["status"]
    new_status = "offline" if current == "online" else "online"
    
    PRESENTATION_NODES[device_id]["status"] = new_status
    PRESENTATION_NODES[device_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
    
    return {"status": "success", "device_id": device_id, "new_status": new_status}

@router.get("/devices/{device_id}")
async def get_device(device_id: str) -> Dict[str, Any]:
    """Returns the registry entry for a single device."""
    try:
        raw_data = await redis_client.get(f"registry:{device_id}")
        if not raw_data:
            raise HTTPException(status_code=404, detail="Device not found in registry")
            
        data = json.loads(raw_data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching device {device_id} from registry: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
