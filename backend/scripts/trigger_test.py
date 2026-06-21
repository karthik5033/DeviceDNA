"""
DeviceDNA — Response Engine Trigger Test
Manually injects a low trust score to fire all tiers of the response engine.
Run from backend/ folder with venv active:
    python -m scripts.trigger_test

This validates:
  1. Tier 2 (rate_limit) triggers at score=65
  2. Tier 3 (sandbox) triggers at score=45
  3. Tier 4 HITL queue triggers at score=25
  4. Tier 5 HITL queue triggers at score=10
  5. response_audit_logs gets written
  6. Redis pending queue is set
"""

import asyncio
import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

TEST_DEVICES = [
    {"device_id": "TEST-TIER2",  "score": 65.0,  "label": "Tier 2 — Rate Limit"},
    {"device_id": "TEST-TIER3",  "score": 45.0,  "label": "Tier 3 — Sandbox"},
    {"device_id": "TEST-TIER4",  "score": 25.0,  "label": "Tier 4 — HITL Quarantine"},
    {"device_id": "TEST-TIER5",  "score":  9.0,  "label": "Tier 5 — HITL Honeypot"},
]

FAKE_SHAP = {
    "device_class": "camera",
    "features": {
        "total_flows": 1200.0,
        "total_bytes": 5242880.0,
        "avg_packet_size": 512.0,
        "external_ratio": 0.92,
        "anomaly_ensemble": 0.88,
        "vae_score": 0.85,
        "gnn_score": 0.91
    }
}

async def run_trigger_tests():
    from app.services.response_engine import response_engine
    from app.db.postgres import AsyncSessionLocal, engine
    from app.db.redis import redis_client
    from sqlalchemy import select, func
    from app.db.models import ResponseAuditLog

    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}  DeviceDNA — Response Engine Trigger Test{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")

    # Clean up any leftover test keys from previous runs
    for dev in TEST_DEVICES:
        did = dev["device_id"]
        for key_pattern in [
            f"response:rate_limit:{did}",
            f"response:sandboxed:{did}",
            f"response:isolated:{did}",
            f"response:honeypot:{did}",
            f"response:pending:{did}",
            f"response:override:{did}",
            f"response:forensic:{did}",
        ]:
            redis_client.delete(key_pattern)
    print(f"  {YELLOW}Cleaned up leftover Redis test keys{RESET}\n")

    # Count audit logs before
    async with AsyncSessionLocal() as session:
        before_count_res = await session.execute(select(func.count()).select_from(ResponseAuditLog))
        before_count = before_count_res.scalar()

    print(f"  Audit log rows before test: {before_count}\n")

    # Run each tier
    for test in TEST_DEVICES:
        did   = test["device_id"]
        score = test["score"]
        label = test["label"]

        print(f"  {BOLD}Testing {label} ({did}, score={score}){RESET}")

        try:
            triggered = await response_engine.evaluate_triggers(
                device_id=did,
                trust_score=score,
                gnn_score=0.91,
                shap_evidence=FAKE_SHAP
            )

            if triggered:
                print(f"    {GREEN}✓ Actions triggered: {triggered}{RESET}")
            else:
                print(f"    {YELLOW}⚠ No actions triggered (may already be active){RESET}")

            # Check Redis state
            redis_keys = {
                "rate_limit": redis_client.exists(f"response:rate_limit:{did}"),
                "sandboxed":  redis_client.exists(f"response:sandboxed:{did}"),
                "isolated":   redis_client.exists(f"response:isolated:{did}"),
                "honeypot":   redis_client.exists(f"response:honeypot:{did}"),
                "pending":    redis_client.exists(f"response:pending:{did}"),
            }
            active = [k for k, v in redis_keys.items() if v]
            print(f"    Redis state: {active if active else 'none'}")

        except Exception as e:
            print(f"    {RED}✗ ERROR: {e}{RESET}")

        await asyncio.sleep(0.2)

    # Count audit logs after
    await asyncio.sleep(0.5)
    async with AsyncSessionLocal() as session:
        after_count_res = await session.execute(select(func.count()).select_from(ResponseAuditLog))
        after_count = after_count_res.scalar()

    print(f"\n  Audit log rows after test:  {after_count}")
    new_rows = after_count - before_count
    if new_rows > 0:
        print(f"  {GREEN}{BOLD}✓ {new_rows} new audit log rows written to PostgreSQL{RESET}")
    else:
        print(f"  {YELLOW}⚠  No new audit rows — Tiers 4/5 only write on approve/deny, not enqueue{RESET}")

    # Show pending HITL queue
    pending_keys = redis_client.keys("response:pending:TEST-*")
    if pending_keys:
        print(f"\n  {CYAN}HITL pending queue ({len(pending_keys)} items):{RESET}")
        for k in pending_keys:
            val = redis_client.get(k)
            if val:
                data = json.loads(val)
                ttl = redis_client.ttl(k)
                print(f"    [{data['device_id']}] action={data['action']} tier={data['target_tier']} score={data['trigger_score']} ttl={ttl}s")
    else:
        print(f"\n  {YELLOW}No HITL pending items (may have expired or not triggered){RESET}")

    # Cleanup test devices
    print(f"\n  {YELLOW}Cleaning up test Redis keys...{RESET}")
    for dev in TEST_DEVICES:
        did = dev["device_id"]
        for key_pattern in [
            f"response:rate_limit:{did}",
            f"response:sandboxed:{did}",
            f"response:isolated:{did}",
            f"response:honeypot:{did}",
            f"response:pending:{did}",
            f"response:override:{did}",
        ]:
            redis_client.delete(key_pattern)

    await engine.dispose()

    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{GREEN}{BOLD}  Trigger test complete!{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_trigger_tests())
