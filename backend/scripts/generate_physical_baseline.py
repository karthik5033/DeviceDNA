import os
import sys
import csv
import random
from datetime import datetime, timedelta, timezone

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.device_profiles import FLEET
from simulator.traffic_generator import generate_flow
from app.services.feature_extraction import extract_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "physical_devices_baseline.csv")

def main():
    print("Initializing Physical Devices Baseline Data Generator...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Filter the physical devices
    physical_devices = [d for d in FLEET if d.get('is_physical')]
    if not physical_devices:
        print("Error: No physical devices found in FLEET configuration!", file=sys.stderr)
        return
        
    print(f"Found {len(physical_devices)} physical devices in configuration:")
    for d in physical_devices:
        print(f"  - {d['id']}: {d['name']} ({d['device_class']})")
        
    # Headers matching InfluxDB schemas and query expectations
    headers = [
        "device_id", "timestamp", "total_flows", "total_bytes", "total_packets",
        "avg_packet_size", "avg_duration_ms", "tcp_ratio", "udp_ratio", "http_ratio",
        "https_ratio", "dns_ratio", "rtsp_ratio", "mqtt_ratio", "hl7_ratio", "modbus_ratio",
        "unique_dst_ips", "unique_dst_ports", "external_ratio"
    ]
    
    records_count = 0
    now = datetime.now(timezone.utc)
    
    print(f"\nGenerating 100 historical records (5-minute windows) for each physical device...")
    
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for dev in physical_devices:
            dev_id = dev['id']
            dev_cls = dev['device_class']
            
            print(f"Generating for {dev_id} ({dev_cls})...")
            
            # Generate 100 sequential 5-minute windows
            for i in range(100):
                # Calculate past timestamp
                # 100 windows * 5 minutes = 500 minutes (~8.3 hours) of baseline history
                ts = now - timedelta(minutes=(99 - i) * 5)
                
                # Simulate normal flow records for this window
                num_flows = random.randint(15, 35) if dev_cls == 'sensor' else random.randint(40, 90)
                flows = [generate_flow(dev) for _ in range(num_flows)]
                
                # Extract feature vector
                features = extract_features(dev_id, dev_cls, flows)
                
                # Initialize protocol fields
                rtsp = 0.0
                mqtt = 0.0
                hl7 = 0.0
                modbus = 0.0
                
                # Map other_protocol_ratio to the correct protocol field
                other_ratio = features.other_protocol_ratio
                if dev_cls == 'camera':
                    rtsp = other_ratio
                elif dev_cls == 'sensor':
                    mqtt = other_ratio
                elif dev_cls == 'medical':
                    hl7 = other_ratio
                elif dev_cls == 'industrial':
                    modbus = other_ratio
                else:
                    # Default fallback for access_control or others
                    rtsp = other_ratio
                    
                # Build CSV row dict
                row = {
                    "device_id": dev_id,
                    "timestamp": ts.isoformat(),
                    "total_flows": features.total_flows,
                    "total_bytes": features.total_bytes,
                    "total_packets": features.total_packets,
                    "avg_packet_size": round(features.avg_packet_size, 4),
                    "avg_duration_ms": round(features.avg_duration_ms, 4),
                    "tcp_ratio": round(features.tcp_ratio, 4),
                    "udp_ratio": round(features.udp_ratio, 4),
                    "http_ratio": round(features.http_ratio, 4),
                    "https_ratio": round(features.https_ratio, 4),
                    "dns_ratio": round(features.dns_ratio, 4),
                    "rtsp_ratio": round(rtsp, 4),
                    "mqtt_ratio": round(mqtt, 4),
                    "hl7_ratio": round(hl7, 4),
                    "modbus_ratio": round(modbus, 4),
                    "unique_dst_ips": features.unique_dst_ips,
                    "unique_dst_ports": features.unique_dst_ports,
                    "external_ratio": round(features.external_traffic_ratio, 4)
                }
                
                writer.writerow(row)
                records_count += 1
                
    print(f"\n[OK] Generation complete! Created {records_count} records.")
    print(f"Data saved to: {CSV_PATH}")
    print("You can open this file to review, modify, or insert your own generated features.")

if __name__ == "__main__":
    main()
