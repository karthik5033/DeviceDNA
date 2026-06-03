import os
import sys
import csv
import asyncio
from datetime import datetime

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from influxdb_client import Point
from simulator.device_profiles import FLEET

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "physical_devices_baseline.csv"
)

# Map device IDs to their classes for tagging
DEVICE_CLASSES = {d['id']: d['device_class'] for d in FLEET}

async def main():
    from app.db.influxdb import InfluxDBService, INFLUXDB_BUCKET, INFLUXDB_ORG
    print("Initializing Physical Devices Baseline Data Importer...")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Baseline CSV file not found at: {CSV_PATH}", file=sys.stderr)
        print("Please run backend/scripts/generate_physical_baseline.py first to create the template.", file=sys.stderr)
        return
        
    # Read the CSV rows
    rows = []
    with open(CSV_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            rows.append(r)
            
    print(f"Read {len(rows)} records from CSV.")
    
    influx_service = InfluxDBService()
    write_api = influx_service.write_api
    
    # Fields that should be floats
    float_fields = [
        "total_flows", "total_bytes", "total_packets", "avg_packet_size",
        "avg_duration_ms", "tcp_ratio", "udp_ratio", "http_ratio",
        "https_ratio", "dns_ratio", "rtsp_ratio", "mqtt_ratio",
        "hl7_ratio", "modbus_ratio", "unique_dst_ips", "unique_dst_ports",
        "external_ratio"
    ]
    
    points = []
    for idx, row in enumerate(rows):
        dev_id = row['device_id']
        dev_class = DEVICE_CLASSES.get(dev_id, 'unknown')
        
        # Parse ISO timestamp
        try:
            # Handle possible trailing Z or offset
            ts_str = row['timestamp']
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(ts_str)
        except ValueError:
            dt = datetime.utcnow()
            
        p = Point("device_features") \
            .tag("device_id", dev_id) \
            .tag("device_class", dev_class) \
            .time(dt)
            
        for f in float_fields:
            if f in row and row[f] != '':
                p.field(f, float(row[f]))
                
        points.append(p)
        
    print(f"Writing {len(points)} feature points to InfluxDB...")
    
    try:
        # Write in batches
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            await write_api.write(bucket=INFLUXDB_BUCKET, record=batch)
            print(f"  Wrote batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
            
        print("[OK] Baseline features successfully imported into InfluxDB!")
    except Exception as e:
        print(f"Error importing baseline data to InfluxDB: {e}", file=sys.stderr)
    finally:
        await influx_service.close()

if __name__ == "__main__":
    # If running on Windows, set correct selector event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
