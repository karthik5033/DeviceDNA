import time
import logging
from app.db.redis import redis_client

logger = logging.getLogger(__name__)

# Key template: response:anomalies:{device_id}
ANOMALY_ZSET_PREFIX = "response:anomalies"
ROLLING_WINDOW_SECS = 3600  # 60 minutes

def record_anomaly_event(device_id: str):
    """
    Pushes the current timestamp to the device's rolling anomaly ZSET in Redis.
    Cleans up old events and refreshes the key TTL.
    """
    try:
        now = time.time()
        key = f"{ANOMALY_ZSET_PREFIX}:{device_id}"
        
        # Add event to Sorted Set: score is timestamp, member is timestamp (or random UUID if unique members needed)
        # Using string representation of timestamp to ensure uniqueness per event
        member = f"{now}-{device_id}"
        redis_client.zadd(key, {member: now})
        
        # Keep ZSET clean: remove everything older than ROLLING_WINDOW_SECS
        cutoff = now - ROLLING_WINDOW_SECS
        redis_client.zremrangebyscore(key, "-inf", cutoff)
        
        # Set TTL to ensure the key eventually expires if no more anomalies happen
        redis_client.expire(key, ROLLING_WINDOW_SECS)
        
        logger.info(f"Recorded anomaly event for device: {device_id}. Timestamp: {now}")
    except Exception as e:
        logger.error(f"Failed to record anomaly event for {device_id}: {e}")

def get_decay_multiplier(device_id: str) -> float:
    """
    Retrieves the decay multiplier based on the count of active anomaly events
    within the last 60 minutes.
    Formula: max(0.40, 1.0 - (event_count * 0.12))
    """
    try:
        now = time.time()
        key = f"{ANOMALY_ZSET_PREFIX}:{device_id}"
        
        # Clean up expired events first
        cutoff = now - ROLLING_WINDOW_SECS
        redis_client.zremrangebyscore(key, "-inf", cutoff)
        
        # Count remaining events
        event_count = redis_client.zcard(key)
        
        # Compute multiplier
        multiplier = max(0.40, 1.0 - (event_count * 0.12))
        if event_count > 0:
            logger.info(f"Device {device_id} has {event_count} active anomalies in 60m window. Decay Multiplier: {multiplier:.4f}")
        return multiplier
    except Exception as e:
        logger.error(f"Failed to get decay multiplier for {device_id}: {e}")
        return 1.0

def decrement_anomaly_event(device_id: str) -> bool:
    """
    Removes the oldest anomaly event from the device's ZSET.
    Used by the Recovery Manager to gradually restore trust.
    Returns True if an event was removed, False if no events were left.
    """
    try:
        key = f"{ANOMALY_ZSET_PREFIX}:{device_id}"
        # Fetch the oldest element (index 0 to 0)
        oldest = redis_client.zrange(key, 0, 0)
        if oldest:
            redis_client.zrem(key, oldest[0])
            logger.info(f"Recovery: Removed oldest anomaly event for {device_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to decrement anomaly event for {device_id}: {e}")
        return False
