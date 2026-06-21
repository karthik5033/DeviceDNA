import logging
import json
import time
from datetime import datetime, timezone

from app.db.redis import redis_client
from app.api.ws import sio
from app.db.postgres import AsyncSessionLocal
from app.db.models import ResponseAuditLog
from app.services.mqtt_dispatcher import mqtt_dispatcher

logger = logging.getLogger(__name__)

# TTL constants (seconds)
ISOLATION_TTL = 3600      # 1 hour
SANDBOX_TTL = 1800        # 30 minutes
RATE_LIMIT_TTL = 1800     # 30 minutes
HONEYPOT_TTL = 3600       # 1 hour
FORENSIC_TTL = 7200       # 2 hours
BLOCK_IP_TTL = 86400      # 24 hours
HITL_QUEUE_TTL = 120      # 2 minutes countdown

class ResponseEngine:
    """
    Enterprise-grade Response Engine for DeviceDNA.
    Implements a 5-Tier risk classification system, automated and HITL-approved responses,
    PostgreSQL audit logging, and MQTT dispatching.
    """

    async def _log_action_to_db(
        self,
        device_id: str,
        trigger_score: float,
        response_tier: int,
        action: str,
        hitl_decision: str,
        notes: str = None,
        shap_evidence: dict = None,
    ):
        """Helper to write response audit records to PostgreSQL."""
        try:
            async with AsyncSessionLocal() as session:
                audit = ResponseAuditLog(
                    device_id=device_id,
                    trigger_score=trigger_score,
                    response_tier=response_tier,
                    action=action,
                    hitl_decision=hitl_decision,
                    notes=notes,
                    shap_evidence=shap_evidence,
                )
                session.add(audit)
                await session.commit()
            logger.info(f"Audit Logged: device={device_id} | action={action} | tier={response_tier} | decision={hitl_decision}")
        except Exception as e:
            logger.error(f"Failed to write ResponseAuditLog for {device_id}: {e}")

    # ── Action Execution Methods ──────────────────────────────────────────────

    async def rate_limit_device(self, device_id: str, score: float, notes: str = None) -> bool:
        """Apply Tier 2: Bandwidth rate limiting (automatic)."""
        key = f"response:rate_limit:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, RATE_LIMIT_TTL, "true")
        await sio.emit("rate_limit_device", {
            "device_id": device_id,
            "action": "rate_limit",
            "timestamp": _utcnow_iso(),
            "score": score
        })
        
        # Publish MQTT command
        mqtt_dispatcher.dispatch_command(device_id, "rate_limit", relay_open=False, rate_delay_ms=500)
        
        await self._log_action_to_db(device_id, score, 2, "rate_limit", "automatic", notes=notes)
        logger.warning(f"⚠️ RESPONSE ACTION [Tier 2] — rate_limit | device={device_id} | score={score:.2f}")
        return True

    async def sandbox_device(self, device_id: str, score: float, notes: str = None) -> bool:
        """Apply Tier 3: Sandboxing / VLAN redirect (automatic)."""
        key = f"response:sandboxed:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, SANDBOX_TTL, "true")
        await sio.emit("sandbox_device", {
            "device_id": device_id,
            "action": "sandbox",
            "timestamp": _utcnow_iso(),
            "score": score
        })
        
        # Publish MQTT command
        mqtt_dispatcher.dispatch_command(device_id, "sandbox", relay_open=False)
        
        await self._log_action_to_db(device_id, score, 3, "sandbox", "automatic", notes=notes)
        logger.warning(f"🔒 RESPONSE ACTION [Tier 3] — sandbox | device={device_id} | score={score:.2f}")
        return True

    async def isolate_device(self, device_id: str, score: float, hitl_decision: str = "automatic") -> bool:
        """Apply Tier 4: Isolation / Quarantine (HITL approved or fallback)."""
        key = f"response:isolated:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, ISOLATION_TTL, "true")
        await sio.emit("isolate_device", {
            "device_id": device_id,
            "action": "isolate",
            "timestamp": _utcnow_iso(),
            "score": score
        })
        
        # Publish MQTT command (Open relay)
        mqtt_dispatcher.dispatch_command(device_id, "quarantine", relay_open=True)
        
        await self._log_action_to_db(device_id, score, 4, "quarantine", hitl_decision)
        logger.warning(f"🚨 RESPONSE ACTION [Tier 4] — quarantine/isolate | device={device_id} | score={score:.2f} | decision={hitl_decision}")
        return True

    async def honeypot_device(self, device_id: str, score: float, hitl_decision: str = "automatic") -> bool:
        """Apply Tier 5: Honeypot redirect (HITL approved or fallback)."""
        key = f"response:honeypot:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, HONEYPOT_TTL, "true")
        await sio.emit("honeypot_device", {
            "device_id": device_id,
            "action": "honeypot",
            "timestamp": _utcnow_iso(),
            "score": score
        })
        
        # Publish MQTT command
        mqtt_dispatcher.dispatch_command(device_id, "honeypot", relay_open=False)
        
        await self._log_action_to_db(device_id, score, 5, "honeypot", hitl_decision)
        logger.warning(f"🍯 RESPONSE ACTION [Tier 5] — honeypot | device={device_id} | score={score:.2f} | decision={hitl_decision}")
        return True

    async def enable_forensic_capture(self, device_id: str) -> bool:
        """Enable full-packet forensic capture."""
        key = f"response:forensic:{device_id}"
        if redis_client.exists(key):
            return False

        redis_client.setex(key, FORENSIC_TTL, "true")
        await sio.emit("forensic_capture", {
            "device_id": device_id,
            "action": "forensic_capture",
            "timestamp": _utcnow_iso()
        })
        logger.warning(f"🔬 FORENSIC ACTION — capture enabled | device={device_id}")
        return True

    # ── Evaluation / Risk Classification ──────────────────────────────────────

    async def evaluate_triggers(
        self,
        device_id: str,
        trust_score: float, # Effective trust score
        gnn_score: float,
        shap_evidence: dict = None,
        previous_trust_score: float = None
    ) -> list[str]:
        """
        Maps the effective trust score to the 5-Tier Risk Classification System.
        Includes Rate of Decline logic: rapid score drops accelerate escalation (PRD 2.2).
        """
        triggered: list[str] = []

        # ── 1. Check for active manual overrides ──────────────────────────────
        override_key = f"response:override:{device_id}"
        if redis_client.exists(override_key):
            logger.info(f"Skipping triggers for {device_id}: Active human override/ignore key present.")
            return triggered

        # ── 2. Run Forensic Capture Triggers (independent of tiers) ──────────
        # Forensic capture if GNN score is extremely elevated
        if gnn_score > 0.85:
            if await self.enable_forensic_capture(device_id):
                triggered.append("enable_forensic_capture")

        # ── 3. Apply 5-Tier Risk Classification mapping ──────────────────────
        
        # Calculate base tier
        base_tier = 1
        if trust_score < 20:
            base_tier = 5
        elif trust_score < 40:
            base_tier = 4
        elif trust_score < 60:
            base_tier = 3
        elif trust_score < 80:
            base_tier = 2

        # Apply Rate of Decline logic (PRD 2.2)
        if previous_trust_score is not None and base_tier > 1:
            drop = previous_trust_score - trust_score
            if drop >= 25.0:
                logger.warning(f"📉 RAPID DECLINE: {device_id} dropped {drop:.1f} pts. Escalating +2 tiers.")
                base_tier = min(5, base_tier + 2)
            elif drop >= 15.0:
                logger.warning(f"📉 FAST DECLINE: {device_id} dropped {drop:.1f} pts. Escalating +1 tier.")
                base_tier = min(5, base_tier + 1)

        # Trigger corresponding action based on final effective tier
        if base_tier == 1:
            pass  # Handled by RecoveryManager
        elif base_tier == 2:
            if await self.rate_limit_device(device_id, trust_score):
                triggered.append("rate_limit_device")
        elif base_tier == 3:
            if await self.sandbox_device(device_id, trust_score):
                triggered.append("sandbox_device")
        elif base_tier == 4:
            action_taken = await self._enqueue_hitl(device_id, 4, "quarantine", trust_score, shap_evidence)
            if action_taken:
                triggered.append("quarantine_pending")
        elif base_tier == 5:
            action_taken = await self._enqueue_hitl(device_id, 5, "honeypot", trust_score, shap_evidence)
            if action_taken:
                triggered.append("honeypot_pending")

        return triggered

    async def _enqueue_hitl(self, device_id: str, tier: int, action: str, score: float, shap_evidence: dict) -> bool:
        """Pushes high-risk actions (Tier 4 & 5) into the HITL Redis pending queue."""
        pending_key = f"response:pending:{device_id}"
        
        # Check if already enqueued or already isolated/honeypotted
        if redis_client.exists(pending_key):
            return False
            
        active_key = f"response:isolated:{device_id}" if tier == 4 else f"response:honeypot:{device_id}"
        if redis_client.exists(active_key):
            return False

        now = time.time()
        expires_at = now + HITL_QUEUE_TTL
        
        payload = {
            "device_id": device_id,
            "target_tier": tier,
            "action": action,
            "trigger_score": score,
            "timestamp": _utcnow_iso(),
            "expires_at": expires_at,
            "shap_evidence": shap_evidence or {}
        }

        # Store in Redis with 120s TTL
        redis_client.setex(pending_key, HITL_QUEUE_TTL, json.dumps(payload))
        
        # Emit WebSocket to dashboard
        await sio.emit("hitl_pending", payload)
        logger.warning(f"⏳ HITL ENQUEUED [Tier {tier}] — {action} pending approval for device={device_id} | score={score:.2f}")
        return True

    async def release_device(self, device_id: str, score: float, hitl_decision: str = "manual_override") -> dict:
        """
        Releases ALL active restrictions for a device (rate_limit, sandbox, isolation, honeypot).
        Dispatches an MQTT 'recover' command and writes an audit record.
        Returns a dict describing which restrictions were active and released.
        """
        released = []
        restriction_map = {
            "rate_limit": f"response:rate_limit:{device_id}",
            "sandbox": f"response:sandboxed:{device_id}",
            "isolation": f"response:isolated:{device_id}",
            "honeypot": f"response:honeypot:{device_id}",
        }
        for name, key in restriction_map.items():
            if redis_client.exists(key):
                redis_client.delete(key)
                released.append(name)

        # Also clear any pending HITL queue and override key
        redis_client.delete(f"response:pending:{device_id}")
        redis_client.delete(f"response:override:{device_id}")

        if released:
            mqtt_dispatcher.dispatch_command(device_id, "recover", relay_open=False)
            await sio.emit("device_released", {
                "device_id": device_id,
                "released": released,
                "timestamp": _utcnow_iso(),
            })
            notes = f"Manual release. Cleared: {', '.join(released)}"
            await self._log_action_to_db(device_id, score, 1, "release", hitl_decision, notes=notes)
            logger.info(f"🔓 RELEASE — device={device_id} | cleared={released} | decision={hitl_decision}")

        return {"device_id": device_id, "released": released}

    # ── Status Query ──────────────────────────────────────────────────────────

    @staticmethod
    def get_device_response_status(device_id: str) -> dict:
        """Read the active response states directly from Redis."""
        rate_limited = redis_client.exists(f"response:rate_limit:{device_id}") == 1
        sandboxed = redis_client.exists(f"response:sandboxed:{device_id}") == 1
        isolated = redis_client.exists(f"response:isolated:{device_id}") == 1
        honeypot = redis_client.exists(f"response:honeypot:{device_id}") == 1
        forensic = redis_client.exists(f"response:forensic:{device_id}") == 1
        override = redis_client.exists(f"response:override:{device_id}") == 1

        pending_raw = redis_client.get(f"response:pending:{device_id}")
        pending = json.loads(pending_raw) if pending_raw else None

        return {
            "device_id": device_id,
            "rate_limited": rate_limited,
            "sandboxed": sandboxed,
            "isolated": isolated,
            "honeypot": honeypot,
            "forensic_capture": forensic,
            "hitl_override_active": override,
            "pending_approval": pending,
            "any_active": any([rate_limited, sandboxed, isolated, honeypot]),
        }


# ── Helper ───────────────────────────────────────────────────────────────────
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

# Singleton instance
response_engine = ResponseEngine()
