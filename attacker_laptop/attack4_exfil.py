import asyncio
import logging
import random
from attacker_laptop.common.config import FLEET, TOPIC_NAME
from attacker_laptop.common.producer import AttackKafkaProducer
from attacker_laptop.common.attack_utils import create_raw_flow

logger = logging.getLogger(__name__)

async def run_exfil_attack(target_id: str = "SIM-0007"):
    """
    Attack 4 - Slow Data Exfiltration
    Behavior: Gradually increases upload size in flows to simulate stealthy data leakage.
    Expected Detector: CUSUM, LSTM
    Expected Trust: slow degradation over time.
    """
    device = next((d for d in FLEET if d["id"] == target_id), None)
    if not device:
        logger.error(f"Target device {target_id} not found in fleet configuration.")
        return

    logger.info(f"🚀 Starting Attack 4 (Slow Exfiltration) targeting device: {target_id} ({device['ip_address']})")
    
    producer = AttackKafkaProducer()
    await producer.start()

    try:
        exfil_ip = "45.33.32.156"  # Mock external exfiltration receiver
        base_bytes = 100 * 1024     # Start at 100KB
        step_bytes = 5 * 1024       # Increment by 5KB
        iteration = 0
        
        while True:
            # Gradually increase bytes and packets with each iteration
            current_bytes = base_bytes + (iteration * step_bytes)
            current_packets = max(10, current_bytes // 1000)
            
            flow = create_raw_flow(
                device_id=device["id"],
                device_class=device["device_class"],
                src_ip=device["ip_address"],
                dst_ip=exfil_ip,
                dst_port=443,
                protocol="HTTPS",
                bytes_count=current_bytes,
                packets_count=current_packets,
                flags="TCP_ACK",
                is_anomalous=True,
                attack_type="slow_exfil"
            )
            
            await producer.send_flow(TOPIC_NAME, flow)
            logger.info(f"Sent Exfil flow ({iteration}): {device['ip_address']} -> {exfil_ip}:443 ({current_bytes / 1024:.1f} KB)")
            
            iteration += 1
            
            # Send every 3 seconds to represent a continuous slow leak
            await asyncio.sleep(3.0)

    except asyncio.CancelledError:
        logger.info("Attack 4 (Exfil) execution stopped.")
    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_exfil_attack())
