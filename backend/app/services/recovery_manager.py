import asyncio
import time
import logging
from app.db.redis import redis_client
from app.api.ws import sio
from app.services.trust_decay import decrement_anomaly_event, get_decay_multiplier
from app.services.mqtt_dispatcher import mqtt_dispatcher

logger = logging.getLogger(__name__)

LAST_ANOMALY_PREFIX = "response:last_anomaly_time"
RECOVERY_WINDOW_SECS = 300  # 5 minutes clean window

async def evaluate_recovery(device_id: str, raw_trust_score: float, effective_trust: float):
    """
    Checks if a device qualifies for trust recovery and restriction release.
    Should be called at the end of each trust scoring cycle.
    """
    try:
        now = time.time()
        last_anomaly_key = f"{LAST_ANOMALY_PREFIX}:{device_id}"
        
        # If the device currently has an anomaly (raw_trust_score < 70), slide the last-anomaly timestamp to now
        # and do not perform any recovery actions.
        if raw_trust_score < 70:
            redis_client.set(last_anomaly_key, str(now))
            return

        # Check if the device is currently under any restriction
        restrictions = {
            "rate_limit": redis_client.exists(f"response:rate_limit:{device_id}") == 1,
            "sandboxed": redis_client.exists(f"response:sandboxed:{device_id}") == 1,
            "isolated": redis_client.exists(f"response:isolated:{device_id}") == 1,
            "honeypot": redis_client.exists(f"response:honeypot:{device_id}") == 1,
        }
        
        is_restricted = any(restrictions.values())
        if not is_restricted:
            return

        # Get last anomaly time
        last_anomaly_val = redis_client.get(last_anomaly_key)
        if not last_anomaly_val:
            # If no last anomaly is stored, set it to now to start the 5-minute counter
            redis_client.set(last_anomaly_key, str(now))
            return

        last_anomaly_time = float(last_anomaly_val)
        time_since_anomaly = now - last_anomaly_time

        if time_since_anomaly >= RECOVERY_WINDOW_SECS:
            # We have had a 5-minute clean window! Trigger gradual recovery step.
            logger.info(f"Device {device_id} has been clean for {time_since_anomaly:.1f}s. Triggering recovery step...")
            
            # Decrement an anomaly event from the ZSET
            removed = decrement_anomaly_event(device_id)
            
            if removed:
                # Reset last anomaly time to start the next 5-minute window for subsequent recovery
                redis_client.set(last_anomaly_key, str(now))
                
                # Check new decay multiplier
                multiplier = get_decay_multiplier(device_id)
                new_effective_trust = raw_trust_score * multiplier
                
                # If all anomalies are cleared (multiplier back to 1.0), release restrictions!
                if multiplier >= 1.0:
                    logger.info(f"Device {device_id} has fully recovered! Releasing all restrictions.")
                    
                    # Delete restriction keys
                    redis_client.delete(f"response:rate_limit:{device_id}")
                    redis_client.delete(f"response:sandboxed:{device_id}")
                    redis_client.delete(f"response:isolated:{device_id}")
                    redis_client.delete(f"response:honeypot:{device_id}")
                    
                    # Dispatch recover command to MQTT
                    mqtt_dispatcher.dispatch_command(device_id, "recover", relay_open=False)
                    
                    # Emit recovery event to Socket.IO (safe async task creation)
                    asyncio.create_task(sio.emit("device_recovered", {
                        "device_id": device_id,
                        "timestamp": datetime_utcnow_iso(),
                        "effective_trust": new_effective_trust
                    }))
                    
                    # Write to database (ResponseAuditLog)
                    from app.db.postgres import AsyncSessionLocal
                    from app.db.models import ResponseAuditLog
                    
                    async def log_recovery():
                        async with AsyncSessionLocal() as session:
                            audit = ResponseAuditLog(
                                device_id=device_id,
                                trigger_score=new_effective_trust,
                                response_tier=1,
                                action="recover",
                                hitl_decision="automatic"
                            )
                            session.add(audit)
                            await session.commit()
                    asyncio.create_task(log_recovery())
                else:
                    # Emit partial recovery event
                    asyncio.create_task(sio.emit("device_recovering", {
                        "device_id": device_id,
                        "timestamp": datetime_utcnow_iso(),
                        "effective_trust": new_effective_trust,
                        "multiplier": multiplier
                    }))

    except Exception as e:
        logger.error(f"Error in evaluate_recovery for {device_id}: {e}")

def datetime_utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
