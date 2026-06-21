import time
import random
import uuid
from datetime import datetime
from simulator.device_profiles import FLEET, DEVICE_PROFILES

# Local cache of active device restrictions, updated by the simulator loop
# Structure: { device_id: { "isolated": bool, "rate_limited": bool, "sandboxed": bool, "honeypot": bool } }
ACTIVE_RESTRICTIONS = {}

# Active attack behaviors injected into devices
ATTACK_STATE = {}

def generate_flow(device):
    """
    Generate a single, realistic network flow record for a given device 
    based on its class profile and active response restrictions.
    """
    device_id = device['id']
    device_class = device['device_class']
    profile = DEVICE_PROFILES[device_class]['normal_behavior']
    
    # Retrieve restrictions
    res = ACTIVE_RESTRICTIONS.get(device_id, {})
    
    # ── Tier 4: Isolation ──
    if res.get("isolated"):
        return None # Zero traffic emitted

    # ── Tier 2: Rate Limit ──
    if res.get("rate_limited"):
        # Drop 80% of traffic
        if random.random() < 0.8:
            return None

    # Check for active attacks (e.g., Recon injection)
    attack = ATTACK_STATE.get(device_id, {})
    is_recon = attack.get("type") == "recon"
    intensity = attack.get("intensity", 0.3)

    if is_recon:
        # Override protocol to TCP, random ports, random internal destinations
        protocol = "TCP"
        dst_port = random.randint(1, 1024) # port entropy
        dst_ip = f"192.168.43.{random.randint(1, 254)}" # unique dst IPs
        is_external = False
        avg_bytes = 64
        packets = 1
        duration_ms = 10
        flags = "TCP_SYN"
    else:
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
            # ── Tier 5: Honeypot ──
            if res.get("honeypot"):
                dst_ip = "10.99.99.99" # Decoy Honeypot IP
            # ── Tier 3: Sandbox ──
            elif res.get("sandboxed"):
                # Sandbox forces all traffic to stay local (internal peers)
                dst_ip = random.choice(device['internal_peers'])
            else:
                dst_ip = random.choice(device['external_peers'])
        else:
            dst_ip = random.choice(device['internal_peers'])
        
        # Packet and byte scaling
        avg_bytes = max(100, int(random.gauss(*profile['avg_bytes_per_flow'])))
        packet_size = profile['packet_size_range']
        # Safe check in case packet_size isn't a tuple/list
        if isinstance(packet_size, (list, tuple)):
            size_min, size_max = packet_size[0], packet_size[1]
        else:
            size_min, size_max = 64, 1500
        p_size = random.randint(size_min, size_max)
        packets = max(1, avg_bytes // p_size)
        duration_ms = random.randint(10, 5000)
        flags = "TCP_ACK" if protocol in ("TCP", "HTTPS") else "NONE"
    
    return {
        "flow_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device_id": device_id,
        "device_class": device_class,
        "src_ip": device['ip_address'],
        "dst_ip": dst_ip,
        "src_port": random.randint(10000, 60000),
        "dst_port": dst_port,
        "protocol": protocol,
        "bytes": avg_bytes,
        "packets": packets,
        "duration_ms": duration_ms,
        "flags": flags,
        "is_anomalous": is_recon
    }

def generate_batch(size=100):
    """Generate a batch of regular traffic flows, filtering out restricted/isolated items."""
    flows = []
    
    # Distribute flows roughly by the expected frequency in profiles
    for _ in range(size):
        # Pick device weighted by their average flow count
        weights = [DEVICE_PROFILES[d['device_class']]['normal_behavior']['avg_flows_per_5min'][0] for d in FLEET]
        device = random.choices(FLEET, weights=weights, k=1)[0]
        flow = generate_flow(device)
        if flow:
            flows.append(flow)
        
    return flows
