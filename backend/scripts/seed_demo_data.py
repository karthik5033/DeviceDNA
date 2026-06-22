import sys
import os
import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta, timezone

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.redis import redis_client
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
from app.db.postgres import AsyncSessionLocal, engine
from app.db.models import Alert
from simulator.device_profiles import FLEET

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "super-secret-influx-token-123")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "devicedna_org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "telemetry")

# Seeder configuration
TOTAL_DEVICES = len(FLEET)
HEALTHY_COUNT = 50
SUSPICIOUS_COUNT = 0
CRITICAL_COUNT = 0

async def seed_data():
    print("Starting DeviceDNA Demo Data Seeder...")
    
    random.seed(42)
    fleet_copy = list(FLEET)
    random.shuffle(fleet_copy)
    
    # All devices are healthy initially
    critical_devices = []
    suspicious_devices = []
    healthy_devices = fleet_copy[:]
    
    now = datetime.now(timezone.utc)
    points_to_write = []
    redis_keys_written = 0
    
    client = InfluxDBClientAsync(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api()
    
    print(f"Generating 24-hour history for {TOTAL_DEVICES} devices...")
    
    for device in FLEET:
        d_id = device['id']
        d_cls = device['device_class']
        
        if device in critical_devices:
            status = 'critical'
            current_score = random.uniform(14, 31)
            baseline = random.uniform(85, 95)
        elif device in suspicious_devices:
            status = 'suspicious'
            current_score = random.uniform(42, 64)
            baseline = random.uniform(80, 90)
        else:
            status = 'healthy'
            current_score = random.uniform(78, 96)
            baseline = current_score
            
        vae = random.uniform(0.01, 0.1) if status == 'healthy' else random.uniform(0.3, 0.8)
        if_sc = random.uniform(0.01, 0.1) if status == 'healthy' else random.uniform(0.4, 0.9)
        lstm = random.uniform(0.01, 0.1) if status == 'healthy' else random.uniform(0.2, 0.7)
        gnn = random.uniform(0.01, 0.1) if status == 'healthy' else random.uniform(0.3, 0.6)
        
        redis_data = {
            "score": current_score,
            "device_id": d_id,
            "device_class": d_cls,
            "timestamp": now.isoformat() + "Z",
            "vae_score": vae,
            "if_score": if_sc,
            "lstm_score": lstm,
            "gnn_score": gnn,
            "ensemble_score": (if_sc * 0.6) + (lstm * 0.2) + (gnn * 0.2),
            "policy_penalty": 0.0 if status == 'healthy' else random.uniform(0.1, 0.5),
            "peer_penalty": 0.0,
            "penalty": 100.0 - current_score
        }
        redis_client.setex(f"trust:{d_id}", 3600, json.dumps(redis_data))
        redis_keys_written += 1
        
        # InfluxDB Data (24 hours at 5-min = 288 points)
        for i in range(288):
            dt = now - timedelta(minutes=(287 - i) * 5)
            
            if status == 'healthy':
                hist_score = baseline + random.uniform(-5, 5)
            elif status == 'suspicious':
                # Downward trend over last 6 hours (last 72 points)
                if i < (288 - 72):
                    hist_score = baseline + random.uniform(-5, 5)
                else:
                    drop_prog = (i - (288 - 72)) / 72.0
                    hist_score = baseline - ((baseline - current_score) * drop_prog) + random.uniform(-2, 2)
            else: # critical
                # Drop in last 2 hours (last 24 points)
                if i < (288 - 24):
                    hist_score = baseline + random.uniform(-5, 5)
                else:
                    drop_prog = (i - (288 - 24)) / 24.0
                    hist_score = baseline - ((baseline - current_score) * drop_prog) + random.uniform(-2, 2)
                    
            hist_score = max(0.0, min(100.0, hist_score))
            
            # Simple inverse relationship for subscores
            h_vae = max(0.0, min(1.0, (100 - hist_score) / 100.0 * random.uniform(0.8, 1.2)))
            h_if = max(0.0, min(1.0, (100 - hist_score) / 100.0 * random.uniform(0.8, 1.2)))
            h_lstm = max(0.0, min(1.0, (100 - hist_score) / 100.0 * random.uniform(0.8, 1.2)))
            h_gnn = max(0.0, min(1.0, (100 - hist_score) / 100.0 * random.uniform(0.8, 1.2)))
            
            p = Point("trust_scores") \
                .tag("device_id", d_id) \
                .tag("device_class", d_cls) \
                .field("trust_score", float(hist_score)) \
                .field("vae_score", float(h_vae)) \
                .field("if_score", float(h_if)) \
                .field("lstm_score", float(h_lstm)) \
                .field("gnn_score", float(h_gnn)) \
                .field("policy_score", 0.0) \
                .field("peer_score", 0.0) \
                .time(dt)
            points_to_write.append(p)
            
    print(f"Writing {len(points_to_write)} points to InfluxDB...")
    batch_size = 2000
    for i in range(0, len(points_to_write), batch_size):
        await write_api.write(bucket=INFLUXDB_BUCKET, record=points_to_write[i:i+batch_size])
        
    await client.close()
    
    print("Generating alerts...")
    alert_configs = [] # Empty by default to keep all devices green
    
    async with AsyncSessionLocal() as session:
        # No alerts to insert initially
        await session.commit()
        
    # We must properly close the postgres engine so the script can exit
    await engine.dispose()
        
    print("\nSeed Complete!")
    print(f"   - Redis Keys Written: {redis_keys_written}")
    print(f"   - InfluxDB Points Written: {len(points_to_write)}")
    print(f"   - PostgreSQL Alerts Inserted: {len(alert_configs)}")

if __name__ == "__main__":
    asyncio.run(seed_data())
