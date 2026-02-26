from typing import List, Dict, Any
from app.schemas.features import FeatureVector
from simulator.device_profiles import KNOWN_EXTERNAL_IPS

def extract_features(device_id: str, device_class: str, flows: List[Dict[str, Any]]) -> FeatureVector:
    """
    Given a list of raw network flows for a specific device, compute 
    the 14-dimensional feature vector over that window of time.
    """
    total_flows = len(flows)
    
    if total_flows == 0:
        # Return zeroed vector for inactive windows
        return FeatureVector(
            device_id=device_id, device_class=device_class,
            total_flows=0, total_bytes=0, total_packets=0,
            avg_packet_size=0.0, avg_duration_ms=0.0,
            tcp_ratio=0.0, udp_ratio=0.0, http_ratio=0.0, https_ratio=0.0, dns_ratio=0.0, other_protocol_ratio=1.0,
            unique_dst_ips=0, unique_dst_ports=0, external_traffic_ratio=0.0
        )

    total_bytes = sum(f.get('bytes', 0) for f in flows)
    total_packets = sum(f.get('packets', 0) for f in flows)
    
    avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0.0
    avg_duration_ms = sum(f.get('duration_ms', 0) for f in flows) / total_flows
    
    # Protocol distribution
    protocols = {'TCP': 0, 'UDP': 0, 'HTTP': 0, 'HTTPS': 0, 'DNS': 0, 'OTHER': 0}
    for f in flows:
        proto = f.get('protocol', 'OTHER')
        if proto in protocols:
            protocols[proto] += 1
        elif f.get('dst_port') == 80:
            protocols['HTTP'] += 1
        elif f.get('dst_port') == 443:
            protocols['HTTPS'] += 1
        elif f.get('dst_port') == 53:
            protocols['DNS'] += 1
        else:
            protocols['OTHER'] += 1

    # Entropy/External calculations
    unique_dst_ips = len(set(f.get('dst_ip') for f in flows))
    unique_dst_ports = len(set(f.get('dst_port') for f in flows))
    
    external_flows = sum(1 for f in flows if f.get('dst_ip') in KNOWN_EXTERNAL_IPS or f.get('dst_ip', '').startswith(('8.', '1.', '104.', '54.', '35.', '151.')))
    
    return FeatureVector(
        device_id=device_id,
        device_class=device_class,
        total_flows=total_flows,
        total_bytes=total_bytes,
        total_packets=total_packets,
        avg_packet_size=float(avg_packet_size),
        avg_duration_ms=float(avg_duration_ms),
        tcp_ratio=protocols['TCP'] / total_flows,
        udp_ratio=protocols['UDP'] / total_flows,
        http_ratio=protocols['HTTP'] / total_flows,
        https_ratio=protocols['HTTPS'] / total_flows,
        dns_ratio=protocols['DNS'] / total_flows,
        other_protocol_ratio=protocols['OTHER'] / total_flows,
        unique_dst_ips=unique_dst_ips,
        unique_dst_ports=unique_dst_ports,
        external_traffic_ratio=external_flows / total_flows
    )
