from pydantic import BaseModel
from typing import Dict

class FeatureVector(BaseModel):
    """
    15-Dimension Feature Vector extracted over a rolling 5-minute window
    of device network telemetry flows.
    """
    device_id: str
    device_class: str
    
    # Volume Metrics
    total_flows: int
    total_bytes: int
    total_packets: int
    avg_packet_size: float
    avg_duration_ms: float
    
    # Protocol Ratios (Sum to 1.0)
    tcp_ratio: float
    udp_ratio: float
    http_ratio: float
    https_ratio: float
    dns_ratio: float
    other_protocol_ratio: float
    
    # Endpoint Entropy metrics
    unique_dst_ips: int
    unique_dst_ports: int
    external_traffic_ratio: float

    def to_tensor_list(self) -> list[float]:
        """Convert features to a flat float list suitable for PyTorch tensor ingestion."""
        return [
            float(self.total_flows),
            float(self.total_bytes),
            float(self.total_packets),
            self.avg_packet_size,
            self.avg_duration_ms,
            self.tcp_ratio,
            self.udp_ratio,
            self.http_ratio,
            self.https_ratio,
            self.dns_ratio,
            self.other_protocol_ratio,
            float(self.unique_dst_ips),
            float(self.unique_dst_ports),
            self.external_traffic_ratio
        ]
        # Total: 14 dimensions representing behavioral state
