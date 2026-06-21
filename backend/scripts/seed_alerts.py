"""
DeviceDNA — Demo Alert Seeder
Seeds realistic-looking security alerts across all device classes.
Run from backend/ folder with venv active:
    python -m scripts.seed_alerts

Safe to run multiple times — uses INSERT OR IGNORE logic via unique IDs.
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime, timedelta
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEMO_ALERTS = [
    # Critical alerts
    {"device_id": "SIM-0007",  "severity": "critical", "trust_score": 18.4, "vae": 0.92, "if_s": 0.88, "lstm": 0.79, "gnn": 0.91, "msg": "Device SIM-0007 trust score critically low (18.4). Possible DDoS source detected.", "minutes_ago": 45},
    {"device_id": "SIM-0046",  "severity": "critical", "trust_score": 22.1, "vae": 0.89, "if_s": 0.91, "lstm": 0.85, "gnn": 0.87, "msg": "Device SIM-0046 trust score critically low (22.1). Lateral movement detected toward medical subnet.", "minutes_ago": 120},
    {"device_id": "mq135_sensor",    "severity": "critical", "trust_score": 14.7, "vae": 0.95, "if_s": 0.93, "lstm": 0.88, "gnn": 0.94, "msg": "Camera mq135_sensor critically anomalous. External RTSP stream detected to unknown IP.", "minutes_ago": 210},
    {"device_id": "SIM-0040",  "severity": "critical", "trust_score": 27.6, "vae": 0.83, "if_s": 0.79, "lstm": 0.81, "gnn": 0.76, "msg": "Device SIM-0040 trust score critically low (27.6). CUSUM drift alarm — data exfiltration pattern.", "minutes_ago": 330},

    # High severity alerts
    {"device_id": "SIM-0015",  "severity": "high", "trust_score": 43.5, "vae": 0.72, "if_s": 0.68, "lstm": 0.61, "gnn": 0.55, "msg": "Device SIM-0015 in high-risk zone (43.5). Anomalous port scan behavior detected.", "minutes_ago": 15},
    {"device_id": "SIM-0021",  "severity": "high", "trust_score": 55.9, "vae": 0.61, "if_s": 0.59, "lstm": 0.55, "gnn": 0.48, "msg": "Device SIM-0021 trust score dropped to high risk (55.9). Unusual dst IP distribution.", "minutes_ago": 28},
    {"device_id": "SIM-0039",  "severity": "high", "trust_score": 54.8, "vae": 0.65, "if_s": 0.62, "lstm": 0.58, "gnn": 0.51, "msg": "Device SIM-0039 trust score dropped sharply by 18.3 points in last cycle.", "minutes_ago": 52},
    {"device_id": "SIM-0042",  "severity": "high", "trust_score": 52.7, "vae": 0.69, "if_s": 0.71, "lstm": 0.63, "gnn": 0.62, "msg": "Device SIM-0042 trust score high risk (52.7). Elevated bytes_sent above class baseline.", "minutes_ago": 67},
    {"device_id": "SIM-0043",  "severity": "high", "trust_score": 59.3, "vae": 0.55, "if_s": 0.49, "lstm": 0.51, "gnn": 0.44, "msg": "Device SIM-0043 sustained high-risk zone. Peer comparison anomaly detected.", "minutes_ago": 90},
    {"device_id": "ldr_sensor", "severity": "high", "trust_score": 48.2, "vae": 0.70, "if_s": 0.67, "lstm": 0.72, "gnn": 0.58, "msg": "Sensor ldr_sensor high risk (48.2). External traffic ratio 0.31 exceeds class maximum of 0.05.", "minutes_ago": 135},
    {"device_id": "SIM-0026",  "severity": "high", "trust_score": 55.9, "vae": 0.58, "if_s": 0.61, "lstm": 0.49, "gnn": 0.53, "msg": "Device SIM-0026 dropped to high risk. GNN lateral movement edge anomaly.", "minutes_ago": 180},
    {"device_id": "SIM-0038",  "severity": "high", "trust_score": 53.3, "vae": 0.67, "if_s": 0.64, "lstm": 0.60, "gnn": 0.57, "msg": "Device SIM-0038 sustained high-risk zone (53.3). LSTM sequence forecast variance exceeded.", "minutes_ago": 225},

    # Medium alerts
    {"device_id": "SIM-0005",  "severity": "medium", "trust_score": 69.8, "vae": 0.35, "if_s": 0.31, "lstm": 0.29, "gnn": 0.22, "msg": "Device SIM-0005 trust score dropped sharply by 16.2 points from 86.0 to 69.8.", "minutes_ago": 8},
    {"device_id": "SIM-0022",  "severity": "medium", "trust_score": 69.2, "vae": 0.38, "if_s": 0.34, "lstm": 0.31, "gnn": 0.19, "msg": "Device SIM-0022 trust score dropped sharply by 17.1 points from 86.3 to 69.2.", "minutes_ago": 35},
    {"device_id": "dht11_sensor",    "severity": "medium", "trust_score": 61.0, "vae": 0.45, "if_s": 0.41, "lstm": 0.38, "gnn": 0.29, "msg": "Camera dht11_sensor medium risk (61.0). External RTSP ratio slightly elevated above normal.", "minutes_ago": 95},
    {"device_id": "ir_sensor","severity": "medium", "trust_score": 64.3, "vae": 0.40, "if_s": 0.37, "lstm": 0.33, "gnn": 0.25, "msg": "Gateway ir_sensor medium risk (64.3). TCP ratio anomaly detected — unusual for class.", "minutes_ago": 160},
    {"device_id": "esp8266_wifi", "severity": "medium", "trust_score": 66.7, "vae": 0.36, "if_s": 0.32, "lstm": 0.28, "gnn": 0.20, "msg": "Motion sensor esp8266_wifi dropped sharply by 15.4 points from 82.1 to 66.7.", "minutes_ago": 270},
]

async def seed_demo_alerts():
    from app.db.postgres import AsyncSessionLocal, engine
    from app.db.models import Alert
    from sqlalchemy import select, func

    print("\n  Seeding demo alerts into PostgreSQL...\n")

    now = datetime.utcnow()
    added = 0

    async with AsyncSessionLocal() as session:
        for spec in DEMO_ALERTS:
            alert_id = f"DEMO-{spec['device_id']}-{spec['severity'][:3].upper()}"
            
            # Check if this demo alert already exists
            existing = await session.execute(
                select(Alert).where(Alert.id == alert_id)
            )
            if existing.scalar():
                print(f"  [SKIP] {alert_id} already exists")
                continue

            ts = now - timedelta(minutes=spec["minutes_ago"])
            alert = Alert(
                id=alert_id,
                device_id=spec["device_id"],
                severity=spec["severity"],
                alert_type="trust_score_drop",
                message=spec["msg"],
                trust_score=spec["trust_score"],
                vae_score=spec["vae"],
                if_score=spec["if_s"],
                lstm_score=spec["lstm"],
                gnn_score=spec["gnn"],
                tib=None,
                timestamp=ts,
                is_resolved=False,
            )
            session.add(alert)
            added += 1
            print(f"  [ADD]  {alert_id} | {spec['severity'].upper()} | score={spec['trust_score']} | {ts.strftime('%H:%M')} ago")

        await session.commit()

    # Final count
    async with AsyncSessionLocal() as session:
        count_res = await session.execute(select(func.count()).select_from(Alert))
        total = count_res.scalar()

    print(f"\n  Added {added} new demo alerts.")
    print(f"  Total alerts in DB: {total}")

    await engine.dispose()
    print("\n  Done!\n")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(seed_demo_alerts())
