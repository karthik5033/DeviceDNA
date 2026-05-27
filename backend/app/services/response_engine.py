"""
Autonomous Response Engine for DeviceDNA.

Provides a structured library of automated response actions (isolate, sandbox,
forensic capture, IP block) that are triggered by threshold rules after each
trust-score evaluation cycle.  Every action is idempotent within its TTL window
— Redis keys are checked before firing so the same action is never spammed on
consecutive 5-second evaluation ticks.
"""

import logging
import json
from datetime import datetime, timezone

from app.db.redis import redis_client
from app.api.ws import sio

logger = logging.getLogger(__name__)

# ── TTL constants (seconds) ──────────────────────────────────────────────────
ISOLATION_TTL   = 3600      # 1 hour
SANDBOX_TTL     = 1800      # 30 minutes
FORENSIC_TTL    = 7200      # 2 hours
BLOCK_IP_TTL    = 86400     # 24 hours


class ResponseEngine:
    """
    Structured response-action library.

    Each public method:
      1. Checks whether the action is already active (Redis key exists).
      2. If not, sets the Redis key with the appropriate TTL.
      3. Emits a Socket.IO event so the SOC dashboard updates in real time.
      4. Logs the action with device_id, action name, and timestamp.
    """

    # ── Action Methods ───────────────────────────────────────────────────────

    async def isolate_device(self, device_id: str) -> bool:
        """
        Network-level device isolation.
        Sets  response:isolated:{device_id} = true   TTL 1 hr
        Emits isolate_device WebSocket event.
        Returns True if the action was newly taken, False if already active.
        """
        key = f"response:isolated:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, ISOLATION_TTL, "true")
        await sio.emit("isolate_device", {
            "device_id": device_id,
            "action": "isolate",
            "timestamp": _utcnow_iso(),
        })
        logger.warning(
            f"🚨 RESPONSE ACTION — isolate_device | device={device_id} | ts={_utcnow_iso()}"
        )
        return True

    async def sandbox_device(self, device_id: str) -> bool:
        """
        Move device into a sandboxed VLAN for observation.
        Sets  response:sandboxed:{device_id} = true   TTL 30 min
        Emits sandbox_device WebSocket event.
        """
        key = f"response:sandboxed:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, SANDBOX_TTL, "true")
        await sio.emit("sandbox_device", {
            "device_id": device_id,
            "action": "sandbox",
            "timestamp": _utcnow_iso(),
        })
        logger.warning(
            f"🔒 RESPONSE ACTION — sandbox_device | device={device_id} | ts={_utcnow_iso()}"
        )
        return True

    async def enable_forensic_capture(self, device_id: str) -> bool:
        """
        Start full-packet forensic capture for the device.
        Sets  response:forensic:{device_id} = true   TTL 2 hr
        Emits forensic_capture WebSocket event.
        """
        key = f"response:forensic:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, FORENSIC_TTL, "true")
        await sio.emit("forensic_capture", {
            "device_id": device_id,
            "action": "forensic_capture",
            "timestamp": _utcnow_iso(),
        })
        logger.warning(
            f"🔬 RESPONSE ACTION — enable_forensic_capture | device={device_id} | ts={_utcnow_iso()}"
        )
        return True

    async def block_ip(self, device_id: str, dst_ip: str) -> bool:
        """
        Firewall-level IP block originating from a specific device context.
        Sets  response:blocked_ip:{dst_ip} = {device_id}   TTL 24 hr
        Emits ip_blocked WebSocket event.
        """
        key = f"response:blocked_ip:{dst_ip}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, BLOCK_IP_TTL, device_id)
        await sio.emit("ip_blocked", {
            "device_id": device_id,
            "dst_ip": dst_ip,
            "action": "block_ip",
            "timestamp": _utcnow_iso(),
        })
        logger.warning(
            f"🛑 RESPONSE ACTION — block_ip | device={device_id} | dst_ip={dst_ip} | ts={_utcnow_iso()}"
        )
        return True

    # ── Automatic Trigger Rules ──────────────────────────────────────────────

    async def evaluate_triggers(
        self,
        device_id: str,
        trust_score: float,
        gnn_score: float,
    ) -> list[str]:
        """
        Apply the automatic response rules *after* the trust engine computes a
        new score.  Rules are evaluated in priority order; each action fires at
        most once per TTL window.

        Returns a list of action names that were newly triggered this cycle.
        """
        triggered: list[str] = []

        # ── Retrieve previous score for delta calculation ────────────────────
        prev_key = f"response:prev_score:{device_id}"
        prev_raw = redis_client.get(prev_key)
        prev_score: float | None = None
        if prev_raw is not None:
            try:
                prev_score = float(prev_raw)
            except (ValueError, TypeError):
                prev_score = None

        # Always store current score for the next cycle's delta computation
        redis_client.set(prev_key, str(trust_score))

        # ── Rule 1: trust_score < 20 → isolate ──────────────────────────────
        if trust_score < 20:
            if await self.isolate_device(device_id):
                triggered.append("isolate_device")

        # ── Rule 2: 20 ≤ trust_score < 40 → sandbox ─────────────────────────
        elif trust_score < 40:
            if await self.sandbox_device(device_id):
                triggered.append("sandbox_device")

        # ── Rule 3: trust drops > 30 points in one cycle → forensic ─────────
        if prev_score is not None and (prev_score - trust_score) > 30:
            if await self.enable_forensic_capture(device_id):
                triggered.append("enable_forensic_capture")

        # ── Rule 4: gnn_score > 0.85 → forensic ─────────────────────────────
        if gnn_score > 0.85:
            if await self.enable_forensic_capture(device_id):
                triggered.append("enable_forensic_capture")

        if triggered:
            logger.info(
                f"Response triggers fired for {device_id}: {triggered} "
                f"(trust={trust_score:.2f}, gnn={gnn_score:.4f})"
            )

        return triggered

    # ── Status Query ─────────────────────────────────────────────────────────

    @staticmethod
    def get_device_response_status(device_id: str) -> dict:
        """
        Read the current active response flags for a device directly from
        Redis.  No database query needed.
        """
        isolated = redis_client.exists(f"response:isolated:{device_id}") == 1
        sandboxed = redis_client.exists(f"response:sandboxed:{device_id}") == 1
        forensic = redis_client.exists(f"response:forensic:{device_id}") == 1

        # Scan for any blocked IPs associated with this device
        blocked_ips: list[str] = []
        cursor = "0"
        while True:
            cursor, keys = redis_client.scan(
                cursor=cursor,
                match="response:blocked_ip:*",
                count=100,
            )
            for key in keys:
                val = redis_client.get(key)
                if val == device_id:
                    # Extract the IP from the key  response:blocked_ip:1.2.3.4
                    ip = key.split("response:blocked_ip:")[-1]
                    blocked_ips.append(ip)
            if cursor == 0 or cursor == "0":
                break

        return {
            "device_id": device_id,
            "isolated": isolated,
            "sandboxed": sandboxed,
            "forensic_capture": forensic,
            "blocked_ips": blocked_ips,
        }


# ── Helper ───────────────────────────────────────────────────────────────────
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# Singleton
response_engine = ResponseEngine()
