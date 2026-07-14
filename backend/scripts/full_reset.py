"""
Full DeviceDNA Reset Script
Flushes ALL cached state from Redis, clears PostgreSQL alerts,
then re-seeds clean healthy data.
"""
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.redis import redis_client
from app.db.postgres import AsyncSessionLocal, engine
from sqlalchemy import text


async def full_reset():
    print("=" * 60)
    print("  DeviceDNA FULL RESET")
    print("=" * 60)

    # ── 1. Nuke ALL relevant Redis keys ──────────────────────────
    print("\n[1/3] Flushing Redis keys...")

    patterns_to_delete = [
        "trust:*",
        "response:anomalies:*",
        "response:isolated:*",
        "response:rate_limit:*",
        "response:sandboxed:*",
        "response:honeypot:*",
        "response:*",
        "alert:dedup:*",
        "attack_state:*",
        "compromised:*",
        "hitl:*",
        "cusum:*",
        "drift:*",
        "recovery:*",
    ]

    total_deleted = 0
    for pattern in patterns_to_delete:
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                redis_client.delete(*keys)
                total_deleted += len(keys)
            if cursor == 0:
                break

    print(f"   Deleted {total_deleted} Redis keys")

    # ── 2. Clear PostgreSQL alerts table ─────────────────────────
    print("\n[2/3] Clearing PostgreSQL alerts...")
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("DELETE FROM alerts"))
            await session.commit()
        print("   Alerts table cleared")
    except Exception as e:
        print(f"   Warning: Could not clear alerts: {e}")

    # ── 3. Re-seed fresh healthy data ────────────────────────────
    print("\n[3/3] Re-seeding fresh healthy data...")

    # Import and run the seeder
    from scripts.seed_demo_data import seed_data
    await seed_data()

    # Close postgres engine
    await engine.dispose()

    print("\n" + "=" * 60)
    print("  RESET COMPLETE — All devices are healthy.")
    print("  Refresh your dashboard (Ctrl+Shift+R) for a clean start.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(full_reset())
