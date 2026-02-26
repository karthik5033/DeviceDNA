import time
import random
import uuid
from datetime import datetime
from simulator.device_profiles import FLEET, DEVICE_PROFILES

def generate_flow(device):
    """
    Generate a single, realistic network flow record for a given device 
    based on its class profile.
    """
    device_class = device['device_class']
    profile = DEVICE_PROFILES[device_class]['normal_behavior']
    
    # Select protocol based on probability distribution
    protocols = list(profile['protocols'].keys())
    probs = list(profile['protocols'].values())
    protocol = random.choices(protocols, weights=probs, k=1)[0]
    dst_port = profile['ports'].get(protocol, random.randint(1024, 65535))
    
    # Internal vs External traffic
    external_ratio = random.gauss(*profile['external_traffic_ratio'])
    external_ratio = max(0.0, min(1.0, external_ratio)) # clamp 0-1
    is_external = random.random() < external_ratio
    
    # Pick destination
    if is_external:
        dst_ip = random.choice(device['external_peers'])
    else:
        dst_ip = random.choice(device['internal_peers'])
    
    # Packet and byte scaling
    avg_bytes = max(100, int(random.gauss(*profile['avg_bytes_per_flow'])))
    packet_size = random.randint(*profile['packet_size_range'])
    packets = max(1, avg_bytes // packet_size)
    
    return {
        "flow_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device_id": device['id'],
        "src_ip": device['ip_address'],
        "dst_ip": dst_ip,
        "src_port": random.randint(10000, 60000),
        "dst_port": dst_port,
        "protocol": protocol,
        "bytes": avg_bytes,
        "packets": packets,
        "duration_ms": random.randint(10, 5000),
        "flags": "TCP_ACK" if protocol == "TCP" or protocol == "HTTPS" else "NONE",
        "is_anomalous": False
    }

def generate_batch(size=100):
    """Generate a batch of regular traffic flows."""
    flows = []
    
    # Distribute flows roughly by the expected frequency in profiles
    for _ in range(size):
        # Pick device weighted by their average flow count
        weights = [DEVICE_PROFILES[d['device_class']]['normal_behavior']['avg_flows_per_5min'][0] for d in FLEET]
        device = random.choices(FLEET, weights=weights, k=1)[0]
        flows.append(generate_flow(device))
        
    return flows
