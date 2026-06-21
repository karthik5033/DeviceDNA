import asyncio
import random
import logging
from attacker_laptop.common.config import FLEET, TOPIC_NAME
from attacker_laptop.common.producer import AttackKafkaProducer
from attacker_laptop.common.attack_utils import create_raw_flow

logger = logging.getLogger(__name__)

async def run_recon_attack(target_id: str = "SIM-0002"):
    """
    Attack 1 - Stealth Reconnaissance
    Behavior: Scans destination IPs and ports using small packets and TCP_SYN flags.
    Expected Detector: VAE, Isolation Forest
    Expected Trust: drops from ~95 to 60.
    """
    device = next((d for d in FLEET if d["id"] == target_id), None)
    if not device:
        logger.error(f"Target device {target_id} not found in fleet configuration.")
        return

    logger.info(f"🚀 Starting Attack 1 (Stealth Recon) targeting device: {target_id} ({device['ip_address']})")
    
    producer = AttackKafkaProducer()
    await producer.start()

    try:
        # Generate scanning flows to multiple mock external IPs and random ports
        external_ips = ["185.220.101.5", "198.51.100.120", "203.0.113.80", "192.0.2.200", "8.8.8.8", "1.1.1.1"]
        ports_pool = [21, 22, 23, 80, 443, 8080, 3389, 445]
        
        while True:
            dst_ip = random.choice(external_ips)
            dst_port = random.choice(ports_pool)
            
            # Recon profile: low byte count, small packets, high unique ports/IPs, external ratio increases
            flow = create_raw_flow(
                device_id=device["id"],
                device_class=device["device_class"],
                src_ip=device["ip_address"],
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol="TCP",
                bytes_count=64, # Small scan packet
                packets_count=1,
                flags="TCP_SYN",
                is_anomalous=True,
                attack_type="stealth_recon"
            )
            
            await producer.send_flow(TOPIC_NAME, flow)
            logger.info(f"Sent Recon flow: {device['ip_address']} -> {dst_ip}:{dst_port} (TCP_SYN)")
            
            # Send flows rapidly to simulate scanning but paced to avoid flooding
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logger.info("Attack 1 (Recon) execution stopped.")
    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_recon_attack())
