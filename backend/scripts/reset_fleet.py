import sys
import os
import json
import logging

# Ensure backend root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.redis import redis_client
from app.services.mqtt_dispatcher import mqtt_dispatcher
from simulator.device_profiles import FLEET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reset_fleet")

def reset_all_devices():
    logger.info("Starting fleet reset sequence...")
    
    # 1. Patterns to delete from Redis
    patterns = [
        "response:rate_limit:*",
        "response:sandboxed:*",
        "response:isolated:*",
        "response:honeypot:*",
        "response:pending:*",
        "response:override:*",
        "response:last_anomaly_time:*",
        "response:anomalies:*",
        "alert:dedup:*",
        "trust:*"
    ]
    
    deleted_count = 0
    for pattern in patterns:
        cursor = "0"
        while True:
            cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                redis_client.delete(*keys)
                deleted_count += len(keys)
            if cursor == 0 or cursor == "0":
                break
                
    logger.info(f"Deleted {deleted_count} restriction and anomaly keys from Redis.")
    
    # 2. Dispatch MQTT 'recover' command to all devices in the fleet
    for d in FLEET:
        d_id = d['id']
        try:
            mqtt_dispatcher.dispatch_command(d_id, "recover", relay_open=False)
            logger.info(f"Sent recover command to device: {d_id}")
        except Exception as e:
            logger.error(f"Failed to send recover command to {d_id}: {e}")
            
    logger.info("Fleet reset sequence completed successfully!")

if __name__ == "__main__":
    reset_all_devices()
