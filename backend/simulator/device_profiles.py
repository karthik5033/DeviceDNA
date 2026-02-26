"""
Device profiles for the DeviceDNA telemetry simulator.
Defines normal behavior for 6 classes of IoT devices.
"""
import random
from typing import Dict, Any, List

# Network Constants
SUBNETS = ["192.168.10.0/24", "192.168.20.0/24", "10.0.5.0/24", "10.0.6.0/24"]
GATEWAYS = ["192.168.10.1", "192.168.20.1", "10.0.5.1", "10.0.6.1"]
KNOWN_EXTERNAL_IPS = [
    "8.8.8.8", "1.1.1.1", "54.239.28.85", "104.244.42.193", "52.94.236.248",
    "3.5.140.206", "35.190.247.0", "151.101.1.69" # Simulating AWS, Google, Cloudflare
]
KNOWN_INTERNAL_SERVERS = ["192.168.10.50", "192.168.10.51", "10.0.5.100"]

# Helper to generate random consistent MAC
def generate_mac():
    return "00:1A:2B:%02x:%02x:%02x" % (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

def generate_ip(vlan: int):
    # Map VLAN to subnet prefix
    subnet = {10: "192.168.10.", 20: "192.168.20.", 30: "10.0.5.", 40: "10.0.6."}.get(vlan, "192.168.1.")
    return f"{subnet}{random.randint(2, 254)}"

# Profiles based on PRD Section 6.1
DEVICE_PROFILES: Dict[str, Dict[str, Any]] = {
    'camera': {
        'count': 12,
        'vlan': 20,
        'normal_behavior': {
            'protocols': {'RTSP': 0.45, 'HTTP': 0.25, 'HTTPS': 0.15, 'DNS': 0.10, 'NTP': 0.05},
            'avg_flows_per_5min': (80, 15),      # (mean, std)
            'avg_bytes_per_flow': (15000, 5000),
            'unique_dst_ips': (3, 1),
            'unique_dst_ports': (4, 1),
            'external_traffic_ratio': (0.15, 0.05),
            'active_hours': (6, 22),             # 6 AM to 10 PM mostly, but streams internally 24/7
            'packet_size_range': (64, 1500),
            'ports': { 'RTSP': 554, 'HTTP': 80, 'HTTPS': 443, 'DNS': 53, 'NTP': 123 }
        }
    },
    'sensor': {
        'count': 10,
        'vlan': 10,
        'normal_behavior': {
            'protocols': {'MQTT': 0.60, 'HTTP': 0.20, 'DNS': 0.15, 'NTP': 0.05},
            'avg_flows_per_5min': (20, 5),
            'avg_bytes_per_flow': (500, 200),
            'unique_dst_ips': (2, 0),            # Gateway + DNS only
            'unique_dst_ports': (2, 0),
            'external_traffic_ratio': (0.01, 0.01), # Almost no external
            'active_hours': (0, 24),              # Always on
            'packet_size_range': (64, 256),
            'ports': { 'MQTT': 1883, 'HTTP': 80, 'HTTPS': 443, 'DNS': 53, 'NTP': 123 }
        }
    },
    'thermostat': {
        'count': 8,
        'vlan': 10,
        'normal_behavior': {
            'protocols': {'HTTPS': 0.50, 'DNS': 0.25, 'NTP': 0.15, 'MQTT': 0.10},
            'avg_flows_per_5min': (15, 5),
            'avg_bytes_per_flow': (1000, 400),
            'unique_dst_ips': (3, 1),
            'unique_dst_ports': (3, 1),
            'external_traffic_ratio': (0.30, 0.10), # Talks to cloud
            'active_hours': (0, 24),
            'packet_size_range': (64, 512),
            'ports': { 'MQTT': 1883, 'HTTP': 80, 'HTTPS': 443, 'DNS': 53, 'NTP': 123 }
        }
    },
    'access_control': {
        'count': 6,
        'vlan': 20,
        'normal_behavior': {
            'protocols': {'HTTPS': 0.40, 'TCP': 0.30, 'DNS': 0.20, 'NTP': 0.10},
            'avg_flows_per_5min': (30, 10),
            'avg_bytes_per_flow': (2000, 800),
            'unique_dst_ips': (4, 1),
            'unique_dst_ports': (3, 1),
            'external_traffic_ratio': (0.10, 0.05),
            'active_hours': (6, 20),
            'packet_size_range': (64, 1024),
            'ports': { 'TCP': 5000, 'HTTP': 80, 'HTTPS': 443, 'DNS': 53, 'NTP': 123 }
        }
    },
    'medical': {
        'count': 8,
        'vlan': 30,
        'normal_behavior': {
            'protocols': {'HL7': 0.35, 'HTTPS': 0.30, 'DICOM': 0.15, 'DNS': 0.10, 'NTP': 0.10},
            'avg_flows_per_5min': (40, 12),
            'avg_bytes_per_flow': (8000, 3000),
            'unique_dst_ips': (5, 2),
            'unique_dst_ports': (4, 1),
            'external_traffic_ratio': (0.05, 0.03), # Mostly internal hospital network
            'active_hours': (0, 24),
            'packet_size_range': (64, 4096),
            'ports': { 'HL7': 2575, 'DICOM': 104, 'HTTPS': 443, 'DNS': 53, 'NTP': 123 }
        }
    },
    'industrial': {
        'count': 6,
        'vlan': 40,
        'normal_behavior': {
            'protocols': {'Modbus': 0.40, 'OPC-UA': 0.25, 'MQTT': 0.15, 'DNS': 0.10, 'NTP': 0.10},
            'avg_flows_per_5min': (50, 15),
            'avg_bytes_per_flow': (3000, 1000),
            'unique_dst_ips': (3, 1),
            'unique_dst_ports': (3, 0),
            'external_traffic_ratio': (0.02, 0.01), # Highly isolated
            'active_hours': (0, 24),
            'packet_size_range': (64, 2048),
            'ports': { 'Modbus': 502, 'OPC-UA': 4840, 'MQTT': 1883, 'DNS': 53, 'NTP': 123 }
        }
    }
}

# Generate 50 devices based on distribution
def generate_fleet():
    fleet = []
    device_id = 1
    
    for device_class, config in DEVICE_PROFILES.items():
        vlan = config['vlan']
        
        for i in range(config['count']):
            device = {
                'id': f"SIM-{device_id:04d}",
                'name': f"{device_class.capitalize()} {i+1}",
                'device_class': device_class,
                'mac_address': generate_mac(),
                'ip_address': generate_ip(vlan),
                'vlan': vlan,
                'status': 'online',
                'internal_peers': random.sample(KNOWN_INTERNAL_SERVERS, 1) + [GATEWAYS[vlan//10 - 1]],
                'external_peers': random.sample(KNOWN_EXTERNAL_IPS, k=random.randint(1, 3))
            }
            fleet.append(device)
            device_id += 1
            
    return fleet

# The master list of 50 seeded devices
FLEET = generate_fleet()
