import os
import json
import logging
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Fetch MQTT Broker connection details from env
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
try:
    MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
except (ValueError, TypeError):
    MQTT_PORT = 1883

# Reconnect settings
_RECONNECT_DELAY_SECS = 5
_MAX_RECONNECT_ATTEMPTS = 10


class MqttCommandDispatcher:
    """
    Publishes JSON command payloads to MQTT topics for IoT device actuation.

    Topics:
      devicedna/{device_id}/command   — targeted command to a single device
      devicedna/broadcast/{action}    — fleet-wide immunization broadcasts
      devicedna/{device_id}/status    — retained status published after each action
    """

    def __init__(self):
        self.client = None
        self._connected = False
        self._reconnect_attempts = 0
        self._init_client()

    def _init_client(self):
        try:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(client_id="devicedna_backend_dispatcher")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            logger.info(f"MQTT dispatcher connecting to {MQTT_HOST}:{MQTT_PORT}")
        except ImportError:
            logger.warning("paho-mqtt not installed — MQTT dispatcher running in simulated mode.")
            self.client = None
        except Exception as e:
            logger.error(
                f"MQTT connection failed ({MQTT_HOST}:{MQTT_PORT}) — running in fallback/simulated mode: {e}"
            )
            self.client = None

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            self._reconnect_attempts = 0
            logger.info(f"MQTT broker connected successfully ({MQTT_HOST}:{MQTT_PORT})")
        else:
            self._connected = False
            logger.warning(f"MQTT connect returned rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT unexpected disconnect (rc={rc}). Scheduling reconnect.")
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Spawn a daemon thread to retry connection without blocking the event loop."""
        def _retry():
            while not self._connected and self._reconnect_attempts < _MAX_RECONNECT_ATTEMPTS:
                self._reconnect_attempts += 1
                logger.info(
                    f"MQTT reconnect attempt {self._reconnect_attempts}/{_MAX_RECONNECT_ATTEMPTS} "
                    f"in {_RECONNECT_DELAY_SECS}s..."
                )
                time.sleep(_RECONNECT_DELAY_SECS)
                try:
                    if self.client:
                        self.client.reconnect()
                        logger.info("MQTT reconnected successfully.")
                        return
                except Exception as e:
                    logger.error(f"MQTT reconnect failed: {e}")
            if not self._connected:
                logger.error(
                    f"MQTT permanently offline after {_MAX_RECONNECT_ATTEMPTS} attempts. "
                    "Falling back to simulated mode."
                )

        t = threading.Thread(target=_retry, daemon=True)
        t.start()

    # ── Command Dispatch ───────────────────────────────────────────────────────

    def dispatch_command(
        self,
        device_id: str,
        action: str,
        relay_open: bool = False,
        rate_delay_ms: int = 0,
        metadata: dict = None,
    ) -> bool:
        """
        Dispatches a command payload to devicedna/{device_id}/command.

        Payload format:
        {
            "device_id": "cam-001",
            "action": "quarantine" | "rate_limit" | "sandbox" | "honeypot" | "recover" | "release",
            "relay_open": true | false,
            "rate_delay_ms": 0,
            "timestamp": "2026-06-20T08:00:00Z",
            "metadata": {}
        }
        """
        payload = {
            "device_id": device_id,
            "action": action,
            "relay_open": relay_open,
            "rate_delay_ms": rate_delay_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        topic = f"devicedna/{device_id}/command"
        payload_str = json.dumps(payload)

        # Also publish a retained status topic so subscribers always see the latest state
        status_topic = f"devicedna/{device_id}/status"
        status_payload = json.dumps({
            "device_id": device_id,
            "action": action,
            "timestamp": payload["timestamp"],
        })

        return self._publish(topic, payload_str, status_topic=status_topic, status_payload=status_payload)

    def publish_broadcast(self, action: str, metadata: dict = None) -> bool:
        """
        Publishes a fleet-wide command to devicedna/broadcast/{action}.
        Used for immunization — tightening thresholds across all devices of a class.

        Payload format:
        {
            "action": "immunize",
            "timestamp": "...",
            "metadata": {"device_class": "camera", "threshold_delta": 0.1}
        }
        """
        payload = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        topic = f"devicedna/broadcast/{action}"
        logger.info(f"MQTT broadcast: {action} → {topic}")
        return self._publish(topic, json.dumps(payload))

    def _publish(
        self,
        topic: str,
        payload_str: str,
        status_topic: str = None,
        status_payload: str = None,
    ) -> bool:
        logger.info(f"MQTT dispatch → {topic} | payload={payload_str}")

        if self.client and self._connected:
            try:
                result = self.client.publish(topic, payload_str, qos=1)
                result.wait_for_publish(timeout=3)
                logger.info(f"MQTT published (mid={result.mid}) → {topic}")

                # Publish retained status if provided
                if status_topic and status_payload:
                    self.client.publish(status_topic, status_payload, qos=0, retain=True)

                return True
            except Exception as e:
                logger.error(f"MQTT publish failed → {topic}: {e}")
                return False
        else:
            # Simulated mode — log as if published
            logger.info(f"[SIMULATED MQTT] → {topic}: {payload_str}")
            return True


# Singleton instance
mqtt_dispatcher = MqttCommandDispatcher()
