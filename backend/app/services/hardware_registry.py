import asyncio
import json
import logging
from datetime import datetime, timezone
import redis.asyncio as aioredis
from app.db.redis import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

# Async Redis client specifically for the device registry
redis_client = aioredis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

PHYSICAL_DEVICES = {
    "HW-001": "sensor",
    "HW-002": "sensor",
    "HW-003": "access_control",
    "HW-004": "industrial",
    "HW-005": "sensor",
    "gateway_01": "access_control",
    "sensor_01": "sensor",
    "motion_01": "access_control",
    "cam_01": "camera",
    "cam_02": "camera"
}

async def mark_seen(device_id: str, device_class: str = None):
    """
    Updates the registry for a device, marking it online and updating its last_seen timestamp.
    Categorizes it as physical or virtual automatically based on device_id.
    """
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Determine source and authoritative class for physical devices
        if device_id in PHYSICAL_DEVICES:
            source = "physical"
            d_class = PHYSICAL_DEVICES[device_id]
        else:
            source = "virtual"
            d_class = device_class or "unknown"
            
        registry_data = {
            "device_id": device_id,
            "device_class": d_class,
            "source": source,
            "last_seen": now_iso,
            "status": "online"
        }
        
        await redis_client.set(f"registry:{device_id}", json.dumps(registry_data))
    except Exception as e:
        logger.error(f"Failed to mark device {device_id} as seen in registry: {e}")

async def check_stale_devices():
    """
    Iterates over the device registry and marks any device offline 
    if its last_seen timestamp is older than 30 seconds.
    """
    try:
        # Fetch all registry keys
        keys = await redis_client.keys("registry:*")
        now = datetime.now(timezone.utc)
        
        for key in keys:
            raw_data = await redis_client.get(key)
            if not raw_data:
                continue
                
            try:
                data = json.loads(raw_data)
                last_seen_str = data.get("last_seen")
                
                if last_seen_str:
                    # Clean the 'Z' if present for fromisoformat compatibility in python 3.9/3.10
                    if last_seen_str.endswith("Z"):
                        last_seen_str = last_seen_str[:-1] + "+00:00"
                    
                    last_seen_dt = datetime.fromisoformat(last_seen_str)
                    diff = (now - last_seen_dt).total_seconds()
                    
                    if diff > 30 and data.get("status") != "offline":
                        data["status"] = "offline"
                        await redis_client.set(key, json.dumps(data))
                        logger.info(f"Registry: Device {data.get('device_id')} marked offline (stale for {diff:.1f}s)")
            except json.JSONDecodeError:
                pass
            except Exception as inner_e:
                logger.error(f"Error processing stale check for registry key {key}: {inner_e}")
                
    except Exception as e:
        logger.error(f"Error executing check_stale_devices: {e}")

async def registry_maintenance_loop():
    """Background task to run check_stale_devices periodically."""
    logger.info("Starting Device Registry maintenance loop (10s interval)...")
    while True:
        await check_stale_devices()
        await asyncio.sleep(10)
