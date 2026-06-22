import asyncio
import json
import os
import random
import sys
import time
import uuid
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiokafka import AIOKafkaProducer
from simulator.device_profiles import FLEET

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
TOPIC_NAME = "raw-flows"

def get_device(device_id):
    return next((d for d in FLEET if d['id'] == device_id), None)

def create_raw_flow(device_id, device_class, src_ip, dst_ip, dst_port, protocol, bytes_count, packets_count, flags="NONE"):
    return {
        "flow_id": str(uuid.uuid4()),
        "timestamp": json.dumps(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())).replace('"', ''),
        "device_id": device_id,
        "device_class": device_class,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(49152, 65535),
        "dst_port": dst_port,
        "protocol": protocol,
        "bytes": bytes_count,
        "packets": packets_count,
        "duration_ms": random.randint(100, 1500),
        "flags": flags,
        "is_anomalous": True
    }

async def run_port_scan(producer, target_id):
    target = get_device(target_id)
    if not target:
        print(f"[-] Error: Target device {target_id} not found.")
        return

    print(f"[*] Simulating Port Scan against {target_id} ({target['ip_address']})...")
    scan_ports = [22, 80, 1883, 5432, 8000]
    attacker_ip = "192.168.43.150" # Lab Attacker IP

    for port in scan_ports:
        for _ in range(5):  # Multiple probes per port
            flow = create_raw_flow(
                device_id=target_id,
                device_class=target['device_class'],
                src_ip=attacker_ip,
                dst_ip=target['ip_address'],
                dst_port=port,
                protocol="TCP",
                bytes_count=64,
                packets_count=1,
                flags="TCP_SYN"
            )
            await producer.send_and_wait(TOPIC_NAME, flow)
        print(f"    [+] Dispatched SYN scan flows to port {port}")
        await asyncio.sleep(0.5)
    print("[+] Port Scan Simulation Complete.")

async def run_beaconing(producer, device_id, c2_ip, duration):
    device = get_device(device_id)
    if not device:
        print(f"[-] Error: Device {device_id} not found.")
        return

    print(f"[*] Simulating Botnet Beaconing from {device_id} to C2 IP {c2_ip}:4444 for {duration}s...")
    end_time = time.time() + duration
    while time.time() < end_time:
        flow = create_raw_flow(
            device_id=device_id,
            device_class=device['device_class'],
            src_ip=device['ip_address'],
            dst_ip=c2_ip,
            dst_port=4444,
            protocol="TCP",
            bytes_count=128,
            packets_count=2,
            flags="TCP_ACK"
        )
        await producer.send_and_wait(TOPIC_NAME, flow)
        print(f"    [+] Beacon flow sent from {device_id} -> {c2_ip}:4444")
        await asyncio.sleep(5)
    print("[+] Beaconing Simulation Complete.")

async def run_exfil(producer, device_id, exfil_ip, duration):
    device = get_device(device_id)
    if not device:
        print(f"[-] Error: Device {device_id} not found.")
        return

    print(f"[*] Simulating Data Exfiltration from {device_id} to {exfil_ip}:9999 for {duration}s...")
    end_time = time.time() + duration
    sizes = [50000, 150000, 500000, 1500000]  # Escalating bytes
    idx = 0
    
    while time.time() < end_time:
        current_size = sizes[min(idx, len(sizes) - 1)]
        flow = create_raw_flow(
            device_id=device_id,
            device_class=device['device_class'],
            src_ip=device['ip_address'],
            dst_ip=exfil_ip,
            dst_port=9999,
            protocol="TCP",
            bytes_count=current_size,
            packets_count=max(10, current_size // 1000),
            flags="TCP_ACK"
        )
        await producer.send_and_wait(TOPIC_NAME, flow)
        print(f"    [+] Dispatched exfil flow: {current_size} bytes to {exfil_ip}:9999")
        idx += 1
        await asyncio.sleep(8)
    print("[+] Data Exfiltration Simulation Complete.")

async def run_lateral(producer, src_id, dst_id):
    src = get_device(src_id)
    dst = get_device(dst_id)
    if not src or not dst:
        print(f"[-] Error: Source {src_id} or Destination {dst_id} not found.")
        return

    print(f"[*] Simulating Lateral Movement from {src_id} -> {dst_id}...")
    ports = [22, 1883, 5432, 80]
    
    for port in ports:
        flow = create_raw_flow(
            device_id=src_id,
            device_class=src['device_class'],
            src_ip=src['ip_address'],
            dst_ip=dst['ip_address'],
            dst_port=port,
            protocol="TCP",
            bytes_count=1024,
            packets_count=10,
            flags="TCP_SYN"
        )
        await producer.send_and_wait(TOPIC_NAME, flow)
        print(f"    [+] Dispatched lateral connection flow to {dst_id} ({dst['ip_address']}) on port {port}")
        await asyncio.sleep(1)
    print("[+] Lateral Movement Simulation Complete.")

async def run_mqtt_flood(producer, src_id, duration):
    src = get_device(src_id)
    if not src:
        print(f"[-] Error: Device {src_id} not found.")
        return

    print(f"[*] Simulating MQTT Telemetry Flood from {src_id} for {duration}s...")
    end_time = time.time() + duration
    broker_ip = "192.168.10.50" # Standard broker IP

    while time.time() < end_time:
        # High frequency, large volume packet simulation
        flow = create_raw_flow(
            device_id=src_id,
            device_class=src['device_class'],
            src_ip=src['ip_address'],
            dst_ip=broker_ip,
            dst_port=1883,
            protocol="MQTT",
            bytes_count=200000, # Large burst size
            packets_count=1200,  # High throughput
            flags="TCP_ACK"
        )
        await producer.send_and_wait(TOPIC_NAME, flow)
        print(f"    [+] Dispatched MQTT Flood flow: 200KB, 1200 packets to broker")
        await asyncio.sleep(2)
    print("[+] MQTT Flood Simulation Complete.")

async def main():
    parser = argparse.ArgumentParser(description="DeviceDNA Safe Threat Flow Simulator")
    parser.add_argument("--attack", choices=["scan", "beacon", "exfil", "lateral", "flood"], required=True, help="Attack scenario type")
    parser.add_argument("--device", required=True, help="Primary device ID (e.g. gateway_01, sensor_01, cam_01)")
    parser.add_argument("--target-device", help="Target device ID for lateral movement scans")
    parser.add_argument("--c2-ip", default="203.0.113.66", help="Configurable C2 IP for beaconing/exfil")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds for loop attacks")
    args = parser.parse_args()

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print(f"[*] Connecting to Kafka Broker at {KAFKA_BROKER}...")
    await producer.start()
    print("[+] Connected successfully.")

    try:
        if args.attack == "scan":
            await run_port_scan(producer, args.device)
        elif args.attack == "beacon":
            await run_beaconing(producer, args.device, args.c2_ip, args.duration)
        elif args.attack == "exfil":
            await run_exfil(producer, args.device, args.c2_ip, args.duration)
        elif args.attack == "lateral":
            if not args.target_device:
                print("[-] Error: --target-device is required for lateral movement simulation.")
                return
            await run_lateral(producer, args.device, args.target_device)
        elif args.attack == "flood":
            await run_mqtt_flood(producer, args.device, args.duration)
    finally:
        await producer.stop()
        print("[*] Kafka producer stopped.")

if __name__ == "__main__":
    asyncio.run(main())
