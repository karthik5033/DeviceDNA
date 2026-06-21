import os
import time
import json
import logging
import asyncio
import threading
import uuid
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP, ICMP
from simulator.device_profiles import FLEET

logger = logging.getLogger(__name__)

# Load fleet mapping
IP_TO_DEVICE = {d['ip_address']: d for d in FLEET}
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
TOPIC_NAME = "raw-flows"

class LivePacketSniffer:
    def __init__(self):
        self.producer = None
        self.active_flows = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        logger.info("Starting Live Packet Sniffer...")
        self.running = True
        
        # Start the background sniffing thread
        self.thread = threading.Thread(target=self._run_sniff, daemon=True)
        self.thread.start()

        # Start background flushing task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._flush_loop())
        except RuntimeError:
            # Fallback if no loop is running yet (will be started by FastAPI lifespan)
            pass

    def _run_sniff(self):
        # Sniff on all interfaces to capture internal/external Docker networking
        try:
            sniff(prn=self._packet_callback, store=0, stop_filter=lambda p: not self.running)
        except Exception as e:
            logger.error(f"Scapy sniff error: {e}")

    def _get_compromised_device_id(self, attack_type):
        try:
            from app.db.redis import redis_client
            for key in redis_client.scan_iter("compromised:*"):
                val = redis_client.get(key)
                if val:
                    data = json.loads(val.decode('utf-8'))
                    if data.get("attack_type") == attack_type or data.get("attack_type") == "coordinated":
                        return key.decode('utf-8').split(":")[1]
        except Exception:
            pass
        return None

    def _packet_callback(self, packet):
        if not IP in packet:
            return

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        src_dev = IP_TO_DEVICE.get(src_ip)
        dst_dev = IP_TO_DEVICE.get(dst_ip)

        # Determine protocol details
        proto = "OTHER"
        src_port = 0
        dst_port = 0
        flags = "NONE"

        if TCP in packet:
            proto = "TCP"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            tcp_flags = packet[TCP].flags
            if tcp_flags & 0x02:  # SYN
                flags = "TCP_SYN"
            elif tcp_flags & 0x10:  # ACK
                flags = "TCP_ACK"
        elif UDP in packet:
            proto = "UDP"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        elif ICMP in packet:
            proto = "ICMP"

        # Resolve compromised source device dynamically
        if not src_dev:
            if dst_port == 4444:
                device_id = self._get_compromised_device_id("beacon")
                if device_id:
                    src_dev = next((d for d in FLEET if d['id'] == device_id), None)
            elif dst_port == 9999:
                device_id = self._get_compromised_device_id("exfil")
                if device_id:
                    src_dev = next((d for d in FLEET if d['id'] == device_id), None)
            elif dst_dev and dst_port in [22, 5432, 1883, 80]:
                device_id = self._get_compromised_device_id("lateral")
                if device_id:
                    src_dev = next((d for d in FLEET if d['id'] == device_id), None)

        if not src_dev and not dst_dev:
            # Not involving our fleet
            return

        pkt_len = len(packet)

        # Record flow for source device if it is in fleet
        if src_dev:
            self._add_packet_to_flow(
                device_id=src_dev['id'],
                device_class=src_dev['device_class'],
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto,
                flags=flags,
                pkt_len=pkt_len
            )

        # Record flow for destination device if it is in fleet (and src is different)
        if dst_dev and src_ip != dst_ip:
            self._add_packet_to_flow(
                device_id=dst_dev['id'],
                device_class=dst_dev['device_class'],
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto,
                flags=flags,
                pkt_len=pkt_len
            )

    def _add_packet_to_flow(self, device_id, device_class, src_ip, dst_ip, src_port, dst_port, protocol, flags, pkt_len):
        key = (device_id, src_ip, dst_ip, src_port, dst_port, protocol)
        with self.lock:
            if key not in self.active_flows:
                self.active_flows[key] = {
                    "bytes": 0,
                    "packets": 0,
                    "flags": "NONE",
                    "start_time": time.time(),
                    "device_class": device_class
                }
            flow = self.active_flows[key]
            flow["bytes"] += pkt_len
            flow["packets"] += 1
            if flags == "TCP_SYN":
                flow["flags"] = "TCP_SYN"
            elif flags == "TCP_ACK" and flow["flags"] == "NONE":
                flow["flags"] = "TCP_ACK"

    async def _flush_loop(self):
        try:
            from aiokafka import AIOKafkaProducer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await self.producer.start()
            logger.info("Sniffer connected to Kafka successfully via AIOKafkaProducer.")

            while self.running:
                await asyncio.sleep(5)
                await self._flush_flows()
        except Exception as e:
            logger.error(f"Sniffer error in flush loop: {e}")
        finally:
            if self.producer:
                await self.producer.stop()
                logger.info("Sniffer AIOKafkaProducer stopped.")

    async def _flush_flows(self):
        with self.lock:
            flows_to_send = self.active_flows
            self.active_flows = {}

        if not flows_to_send:
            return

        now_str = datetime.utcnow().isoformat() + "Z"
        sent_count = 0
        for key, flow_data in flows_to_send.items():
            device_id, src_ip, dst_ip, src_port, dst_port, protocol = key
            duration = int((time.time() - flow_data["start_time"]) * 1000)
            
            flow_record = {
                "flow_id": str(uuid.uuid4()),
                "timestamp": now_str,
                "device_id": device_id,
                "device_class": flow_data["device_class"],
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "bytes": flow_data["bytes"],
                "packets": flow_data["packets"],
                "duration_ms": max(1, duration),
                "flags": flow_data["flags"],
                "is_anomalous": False
            }

            if self.producer:
                try:
                    await self.producer.send(TOPIC_NAME, flow_record)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Sniffer failed to send flow to Kafka: {e}")
        
        if sent_count > 0:
            logger.info(f"Sniffer sent {sent_count} aggregated flows to Kafka.")

    def stop(self):
        self.running = False
        logger.info("Live Packet Sniffer stopped.")

# Singleton instance
live_sniffer = LivePacketSniffer()
