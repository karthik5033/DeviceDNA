import os
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
import logging

logger = logging.getLogger(__name__)

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "super-secret-influx-token-123")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "devicedna_org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "telemetry")

class InfluxDBService:
    def __init__(self):
        self.client = InfluxDBClientAsync(
            url=INFLUXDB_URL, 
            token=INFLUXDB_TOKEN, 
            org=INFLUXDB_ORG
        )
        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()

    async def write_flow(self, flow_data: dict):
        """Write a raw flow reading into InfluxDB Async."""
        try:
            point = Point("raw_flows") \
                .tag("device_id", flow_data['device_id']) \
                .tag("src_ip", flow_data['src_ip']) \
                .tag("dst_ip", flow_data['dst_ip']) \
                .tag("protocol", flow_data['protocol']) \
                .field("bytes", flow_data['bytes']) \
                .field("packets", flow_data['packets']) \
                .field("duration_ms", flow_data['duration_ms']) \
                .field("src_port", flow_data['src_port']) \
                .field("dst_port", flow_data['dst_port']) \
                .field("flags", flow_data['flags'])

            await self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        except Exception as e:
            logger.error(f"Failed to write flow to InfluxDB: {e}")

    async def write_feature_vector(self, device_id: str, device_class: str, features: dict):
        """Write normalized 5-min feature vectors."""
        try:
            point = Point("device_features") \
                .tag("device_id", device_id) \
                .tag("device_class", device_class)
                
            for k, v in features.items():
                point.field(k, v)

            await self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        except Exception as e:
            logger.error(f"Failed to write feature vector to InfluxDB: {e}")

    async def write_trust_score(self, device_id: str, device_class: str, payload: dict):
        """Write the computed trust score and subscores to InfluxDB Async."""
        try:
            point = Point("trust_scores") \
                .tag("device_id", device_id) \
                .tag("device_class", device_class) \
                .field("trust_score", float(payload.get("score", 0.0))) \
                .field("vae_score", float(payload.get("vae_score", 0.0))) \
                .field("if_score", float(payload.get("if_score", 0.0))) \
                .field("lstm_score", float(payload.get("lstm_score", 0.0))) \
                .field("gnn_score", float(payload.get("gnn_score", 0.0))) \
                .field("policy_score", float(payload.get("policy_penalty", 0.0))) \
                .field("peer_score", float(payload.get("peer_penalty", 0.0)))

            await self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        except Exception as e:
            logger.error(f"Failed to write trust score to InfluxDB: {e}")

    async def query_trust_history(self, device_id: str, hours: int = 24):
        """Query trust score history for a specific device."""
        try:
            query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: -{hours}h)
                |> filter(fn: (r) => r["_measurement"] == "trust_scores")
                |> filter(fn: (r) => r["device_id"] == "{device_id}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            result = await self.query_api.query(query=query, org=INFLUXDB_ORG)
            
            history = []
            for table in result:
                for record in table.records:
                    dt = record.get_time()
                    history.append({
                        "timestamp": dt.isoformat() if dt else None,
                        "trust_score": record.values.get("trust_score"),
                        "vae_score": record.values.get("vae_score"),
                        "if_score": record.values.get("if_score"),
                        "lstm_score": record.values.get("lstm_score"),
                        "gnn_score": record.values.get("gnn_score"),
                        "policy_score": record.values.get("policy_score"),
                        "peer_score": record.values.get("peer_score")
                    })
            # Ensure chronological order (oldest to newest)
            history.sort(key=lambda x: x["timestamp"] if x["timestamp"] else "")
            return history
        except Exception as e:
            logger.error(f"Failed to query trust history from InfluxDB: {e}")
            return []

    async def query_feature_history(self, device_id: str, days: int = 7):
        """Query historical 14D feature vectors for a specific device to train on observed behavior."""
        try:
            query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: -{days}d)
                |> filter(fn: (r) => r["_measurement"] == "device_features")
                |> filter(fn: (r) => r["device_id"] == "{device_id}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            result = await self.query_api.query(query=query, org=INFLUXDB_ORG)
            
            history = []
            for table in result:
                for record in table.records:
                    # Collect all 14 fields
                    features = {
                        "total_flows": record.values.get("total_flows", 0.0),
                        "total_bytes": record.values.get("total_bytes", 0.0),
                        "total_packets": record.values.get("total_packets", 0.0),
                        "avg_packet_size": record.values.get("avg_packet_size", 0.0),
                        "avg_duration_ms": record.values.get("avg_duration_ms", 0.0),
                        "tcp_ratio": record.values.get("tcp_ratio", 0.0),
                        "udp_ratio": record.values.get("udp_ratio", 0.0),
                        "http_ratio": record.values.get("http_ratio", 0.0),
                        "https_ratio": record.values.get("https_ratio", 0.0),
                        "dns_ratio": record.values.get("dns_ratio", 0.0),
                        "other_ratio": record.values.get("rtsp_ratio", 0.0) + record.values.get("mqtt_ratio", 0.0) + record.values.get("hl7_ratio", 0.0) + record.values.get("modbus_ratio", 0.0),
                        "unique_dst_ips": record.values.get("unique_dst_ips", 0.0),
                        "unique_dst_ports": record.values.get("unique_dst_ports", 0.0),
                        "external_ratio": record.values.get("external_ratio", 0.0)
                    }
                    history.append(features)
            return history
        except Exception as e:
            logger.error(f"Failed to query feature history from InfluxDB: {e}")
            return []

    async def close(self):
        await self.client.close()

# Singleton Instance
influx_db = InfluxDBService()
