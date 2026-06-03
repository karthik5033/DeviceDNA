import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Fetch MQTT Broker connection details from env
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
try:
    MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
except (ValueError, TypeError):
    MQTT_PORT = 1883

class MqttCommandDispatcher:
    """
    Publishes JSON command payloads to MQTT topic devicedna/{device_id}/command
    to actuate responses on physical or simulated IoT devices.
    """
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            import paho.mqtt.client as mqtt
            # Use clean, modern client initialization
            self.client = mqtt.Client(client_id="devicedna_backend_dispatcher")
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        except ImportError:
            logger.warning("paho-mqtt is not installed. MQTT command dispatch will run in simulated mode.")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker at {MQTT_HOST}:{MQTT_PORT} (Running in fallback mode): {e}")
            self.client = None

    def dispatch_command(self, device_id: str, action: str, relay_open: bool = False, rate_delay_ms: int = 0):
        """
        Dispatches a command payload to devicedna/{device_id}/command.
        Payload format:
        {
            "device_id": "SIM-0001",
            "action": "quarantine" | "rate_limit" | "sandbox" | "honeypot" | "recover",
            "relay_open": true | false,
            "rate_delay_ms": 0,
            "timestamp": "2026-05-31T06:40:00Z"
        }
        """
        payload = {
            "device_id": device_id,
            "action": action,
            "relay_open": relay_open,
            "rate_delay_ms": rate_delay_ms,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }
        
        topic = f"devicedna/{device_id}/command"
        payload_str = json.dumps(payload)

        logger.info(f"Dispatching MQTT command: {payload_str} to topic {topic}")

        if self.client:
            try:
                self.client.publish(topic, payload_str, qos=1)
                logger.info(f"MQTT Command successfully published to {topic}")
                return True
            except Exception as e:
                logger.error(f"Failed to publish MQTT command to {topic}: {e}")
                return False
        else:
            logger.info(f"[SIMULATED MQTT] Published command to {topic}: {payload_str}")
            return True

# Singleton instance
mqtt_dispatcher = MqttCommandDispatcher()
