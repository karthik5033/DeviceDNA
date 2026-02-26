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

    async def close(self):
        await self.client.close()

# Singleton Instance
influx_db = InfluxDBService()
