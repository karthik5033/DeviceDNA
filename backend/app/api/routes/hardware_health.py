from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
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

@router.get("/devices")
async def get_all_devices() -> List[Dict[str, Any]]:
    """Returns the full device registry from Redis."""
    try:
        keys = await redis_client.keys("registry:*")
        devices = []
        for key in keys:
            raw_data = await redis_client.get(key)
            if raw_data:
                try:
                    data = json.loads(raw_data)
                    devices.append(data)
                except json.JSONDecodeError:
                    continue
        
        # Sort by device_id
        devices.sort(key=lambda x: x.get("device_id", ""))
        return devices
    except Exception as e:
        logger.error(f"Error fetching hardware registry: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching registry")

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
