import uuid
import random
from datetime import datetime

def create_raw_flow(
    device_id: str,
    device_class: str,
    src_ip: str,
    dst_ip: str,
    dst_port: int,
    protocol: str,
    bytes_count: int,
    packets_count: int,
    flags: str = "NONE",
    is_anomalous: bool = False,
    attack_type: str = None
) -> dict:
    """
    Creates a dictionary representation of a raw flow matching the backend's expected schema.
    """
    return {
        "flow_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device_id": device_id,
        "device_class": device_class,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(10240, 65535),
        "dst_port": dst_port,
        "protocol": protocol,
        "bytes": bytes_count,
        "packets": packets_count,
        "duration_ms": random.randint(10, 2000),
        "flags": flags,
        "is_anomalous": is_anomalous,
        "attack_type": attack_type
    }
