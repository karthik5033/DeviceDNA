import random
import uuid
from datetime import datetime
from simulator.device_profiles import FLEET

class AttackScenarios:
    """Four distinct attack scenarios from PRD section 6.3."""
    
    @staticmethod
    def scenario_1_botnet_c2():
        """
        Target: A Camera (SIM-0014)
        Behavior: Sudden connections to 3 new external IPs. C2 beaconing.
        Detected By: VAE + Isolation Forest (Hard Anomaly)
        """
        camera = next(d for d in FLEET if d['device_class'] == 'camera')
        
        # Anomalous behavior: New IPs, unknown port, periodic small packets
        return {
            "flow_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device_id": camera['id'],
            "src_ip": camera['ip_address'],
            "dst_ip": random.choice(["203.0.113.4", "198.51.100.22", "192.0.2.77"]), # Malicious C2 IPs
            "src_port": random.randint(30000, 50000),
            "dst_port": 4444, # Anomalous port
            "protocol": "TCP",
            "bytes": 128,     # Small beacon payload
            "packets": 2,
            "duration_ms": 1500,
            "flags": "TCP_SYN",
            "is_anomalous": True,
            "attack_type": "botnet_c2"
        }

    @staticmethod
    def scenario_2_slow_exfiltration():
        """
        Target: A Sensor (SIM-0007)
        Behavior: Upload volume increases slightly but steadily over long period.
        Detected By: CUSUM Drift detection (Soft Drift)
        """
        sensor = next(d for d in FLEET if d['device_class'] == 'sensor')
        
        # Volumetric anomaly over time. Much larger byte count than the 500 avg.
        return {
            "flow_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device_id": sensor['id'],
            "src_ip": sensor['ip_address'],
            "dst_ip": random.choice(sensor['external_peers'] + ["45.33.32.156"]),
            "src_port": random.randint(10000, 60000),
            "dst_port": 443,
            "protocol": "HTTPS",
            "bytes": random.randint(5000, 8000), # Significantly larger exfil payload
            "packets": random.randint(10, 20),
            "duration_ms": 5000,
            "flags": "TCP_ACK",
            "is_anomalous": True,
            "attack_type": "slow_exfil"
        }

    @staticmethod
    def scenario_3_lateral_movement():
        """
        Target: 3 Medical Devices
        Behavior: Devices suddenly communicating internally where no edge previously existed.
        Detected By: GNN / GraphSAGE (Topological anomaly)
        """
        medical_devices = [d for d in FLEET if d['device_class'] == 'medical'][:3]
        
        source = medical_devices[0]
        target = medical_devices[1]
        
        # Normal-looking packet, but totally anomalous topological edge
        return {
            "flow_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device_id": source['id'],
            "src_ip": source['ip_address'],
            "dst_ip": target['ip_address'],
            "src_port": random.randint(10000, 60000),
            "dst_port": 22, # SSH attempt internally
            "protocol": "TCP",
            "bytes": 2048,
            "packets": 10,
            "duration_ms": 200,
            "flags": "TCP_SYN",
            "is_anomalous": True,
            "attack_type": "lateral_movement"
        }

    @staticmethod
    def scenario_4_nlp_policy_trigger():
        """
        Target: Thermostat
        Behavior: Contacts TOR exit node.
        Detected By: Policy Engine (Triggered by plain-english rule)
        """
        thermostat = next(d for d in FLEET if d['device_class'] == 'thermostat')
        
        # Policy explicitly forbids TOR exit nodes
        tor_exit_node = "185.220.101.43" 
        
        return {
            "flow_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device_id": thermostat['id'],
            "src_ip": thermostat['ip_address'],
            "dst_ip": tor_exit_node,
            "src_port": random.randint(10000, 60000),
            "dst_port": 443,
            "protocol": "HTTPS",
            "bytes": 500,
            "packets": 4,
            "duration_ms": 100,
            "flags": "TCP_ACK",
            "is_anomalous": True,
            "attack_type": "policy_violation"
        }
