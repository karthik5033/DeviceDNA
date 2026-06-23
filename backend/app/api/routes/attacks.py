"""
Attack Trigger API
==================
Allows the dashboard to start and stop attack simulations without
running the CLI scripts manually. Writes attack_state:<device_id>
keys to Redis — the simulator's traffic_generator.py reads these
every cycle and generates matching anomalous flows.
"""

import json
import asyncio
import threading
import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from app.db.redis import redis_client

router = APIRouter(prefix="/api/attack", tags=["Attack Simulation"])

# ---------------------------------------------------------------------------
# Device targets per attack
# ---------------------------------------------------------------------------

ATTACK_TARGETS = {
    1: {
        "name": "Stealth Recon Scan",
        "description": "Port-scanning recon injected into cameras and sensors. High unique-dst-IPs, TCP SYN flood, low packet size.",
        "targets": ["dht11_sensor", "mq135_sensor", "ir_sensor", "SIM-0005", "SIM-0015", "SIM-0030", "esp8266_wifi"],
        "payload": {"type": "recon", "intensity": 0.3},
        "duration_default": 300,
    },
    2: {
        "name": "Two-Stage Botnet C2 + DDoS",
        "description": "Stage 1: 120s of C2 beaconing on port 4444. Stage 2: 180s volumetric UDP DDoS flood.",
        "targets": ["SIM-0009", "SIM-0012", "dht11_sensor", "SIM-0003", "esp8266_wifi"],
        "payload": {
            "type": "beacon",
            "intensity": 0.7,
            "c2_servers": ["203.0.113.4", "198.51.100.22", "192.0.2.77"],
            "c2_port": 4444,
            "beacon_interval_sec": 30,
            "beacon_payload_bytes": 128,
        },
        "duration_default": 300,
    },
    3: {
        "name": "Lateral Movement / Worm Spread",
        "description": "Simulates internal lateral movement by aggressively scanning internal peers on SSH/SMB/RDP ports.",
        "targets": ["SIM-0010", "SIM-0011", "SIM-0016", "esp8266_wifi"],
        "payload": {"type": "lateral", "intensity": 0.8},
        "duration_default": 300,
    },
    4: {
        "name": "Massive Data Exfiltration",
        "description": "Simulates ransomware or spyware exfiltrating huge volumes of outbound data over HTTPS.",
        "targets": ["SIM-0040", "SIM-0041", "SIM-0045", "esp8266_wifi"],
        "payload": {"type": "exfil", "intensity": 1.0},
        "duration_default": 300,
    },
}

# Track running attack threads so we can cancel them
_active_attack: dict = {}


class TriggerRequest(BaseModel):
    attack_id: int
    duration: Optional[int] = None  # Override default duration (seconds)


def _inject_keys(targets: list, payload: dict, ttl_seconds: int):
    """Write attack_state keys into Redis with a TTL so they auto-expire."""
    for device_id in targets:
        redis_client.setex(f"attack_state:{device_id}", ttl_seconds, json.dumps(payload))


def _clear_keys(targets: list):
    """Delete all attack_state keys for the given targets."""
    for device_id in targets:
        redis_client.delete(f"attack_state:{device_id}")


def _run_attack2_stages(targets: list, beacon_payload: dict, ddos_duration: int, beacon_duration: int):
    """
    Background thread for Attack 2 two-stage logic:
      Stage 1 (beacon_duration sec) — C2 beaconing
      Stage 2 (ddos_duration sec)   — Volumetric DDoS
    """
    try:
        # Stage 1: Beacon
        _inject_keys(targets, beacon_payload, beacon_duration + ddos_duration + 60)
        time.sleep(beacon_duration)

        # Check if attack was cancelled between stages
        if not _active_attack.get("running"):
            return

        # Stage 2: Upgrade to DDoS
        ddos_payload = {
            "type": "ddos",
            "intensity": 1.0,
            "ddos_target_ip": "198.51.100.99",
        }
        _inject_keys(targets, ddos_payload, ddos_duration + 60)
        time.sleep(ddos_duration)

    finally:
        _clear_keys(targets)
        _active_attack["running"] = False
        _active_attack["attack_id"] = None


def _run_attack1_stages(targets: list, payload: dict, duration: int):
    """Background thread for Attack 1 (simple single-stage recon)."""
    try:
        _inject_keys(targets, payload, duration + 60)
        time.sleep(duration)
    finally:
        _clear_keys(targets)
        _active_attack["running"] = False
        _active_attack["attack_id"] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_attack_status():
    """
    Returns the current attack simulation state.
    Reads active attack_state:* keys from Redis to report live status.
    """
    active_targets = {}
    try:
        for key in redis_client.scan_iter("attack_state:*"):
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            device_id = key_str.split("attack_state:")[1]
            val = redis_client.get(key)
            if val:
                active_targets[device_id] = json.loads(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis scan failed: {e}")

    return {
        "is_running": _active_attack.get("running", False),
        "attack_id": _active_attack.get("attack_id"),
        "started_at": _active_attack.get("started_at"),
        "active_targets": active_targets,
    }


@router.post("/trigger")
async def trigger_attack(req: TriggerRequest, background_tasks: BackgroundTasks):
    """
    Trigger an attack simulation by ID.
    
    - attack_id=1 → Stealth Recon Scan (300s default)
    - attack_id=2 → Two-Stage Botnet C2 + DDoS (300s default: 120s beacon + 180s ddos)
    """
    if _active_attack.get("running"):
        raise HTTPException(
            status_code=409,
            detail=f"Attack {_active_attack.get('attack_id')} is already running. Stop it first.",
        )

    if req.attack_id not in ATTACK_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid attack_id {req.attack_id}. Valid: {list(ATTACK_TARGETS.keys())}",
        )

    spec = ATTACK_TARGETS[req.attack_id]
    total_duration = req.duration or spec["duration_default"]
    targets = spec["targets"]

    _active_attack["running"] = True
    _active_attack["attack_id"] = req.attack_id
    _active_attack["started_at"] = time.time()

    if req.attack_id == 2:
        # Two-stage: 40% beacon, 60% ddos
        beacon_duration = min(120, int(total_duration * 0.4))
        ddos_duration = total_duration - beacon_duration
        t = threading.Thread(
            target=_run_attack2_stages,
            args=(targets, spec["payload"], ddos_duration, beacon_duration),
            daemon=True,
        )
    else:
        t = threading.Thread(
            target=_run_attack1_stages,
            args=(targets, spec["payload"], total_duration),
            daemon=True,
        )

    t.start()

    return {
        "status": "started",
        "attack_id": req.attack_id,
        "name": spec["name"],
        "targets": targets,
        "total_duration_seconds": total_duration,
        "description": spec["description"],
    }


@router.post("/stop")
async def stop_attack():
    """
    Immediately terminate any running attack by deleting all attack_state Redis keys.
    """
    _active_attack["running"] = False

    # Clear all attack_state keys for all known targets
    cleared = []
    for spec in ATTACK_TARGETS.values():
        for device_id in spec["targets"]:
            if redis_client.exists(f"attack_state:{device_id}"):
                redis_client.delete(f"attack_state:{device_id}")
                cleared.append(device_id)

    # Also do a wildcard scan to catch any stragglers
    try:
        for key in redis_client.scan_iter("attack_state:*"):
            redis_client.delete(key)
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            device_id = key_str.split("attack_state:")[1]
            if device_id not in cleared:
                cleared.append(device_id)
    except Exception:
        pass

    from app.services.response_engine import response_engine
    for device_id in set(cleared):
        try:
            await response_engine.release_device(device_id, score=100.0, hitl_decision="manual_override")
        except Exception:
            pass

    return {
        "status": "stopped",
        "cleared_devices": list(set(cleared)),
    }


@router.get("/list")
async def list_attacks():
    """List all available attack simulations with their descriptions and targets."""
    return [
        {
            "id": aid,
            "name": spec["name"],
            "description": spec["description"],
            "targets": spec["targets"],
            "default_duration_seconds": spec["duration_default"],
        }
        for aid, spec in ATTACK_TARGETS.items()
    ]
