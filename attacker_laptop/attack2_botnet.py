import asyncio
import logging
from attacker_laptop.common.config import FLEET, TOPIC_NAME
from attacker_laptop.common.producer import AttackKafkaProducer
from attacker_laptop.common.attack_utils import create_raw_flow

logger = logging.getLogger(__name__)

async def run_botnet_attack(target_id: str = "SIM-0010"):
    """
    Attack 2 - Botnet C2 Beaconing
    Behavior: Periodic connection to 1 external malicious IP on anomalous port with constant packet size.
    Expected Detector: VAE, LSTM
    Expected Trust: drops from ~95 to 30.
    """
    device = next((d for d in FLEET if d["id"] == target_id), None)
    if not device:
        logger.error(f"Target device {target_id} not found in fleet configuration.")
        return

    logger.info(f"🚀 Starting Attack 2 (Botnet C2 Beaconing) targeting device: {target_id} ({device['ip_address']})")
    
    producer = AttackKafkaProducer()
    await producer.start()

    try:
        c2_ip = "203.0.113.66"  # Malicious C2 IP
        c2_port = 4444          # Anomalous port
        
        while True:
            # Botnet C2 profile: regular interval, static packet size, external target
            flow = create_raw_flow(
                device_id=device["id"],
                device_class=device["device_class"],
                src_ip=device["ip_address"],
                dst_ip=c2_ip,
                dst_port=c2_port,
                protocol="TCP",
                bytes_count=128, # Small repetitive beacon payload
                packets_count=2,
                flags="TCP_ACK",
                is_anomalous=True,
                attack_type="botnet_c2"
            )
            
            await producer.send_flow(TOPIC_NAME, flow)
            logger.info(f"Sent Botnet beacon flow: {device['ip_address']} -> {c2_ip}:{c2_port} (TCP)")
            
            # Regular beaconing interval (every 1.5 seconds)
            await asyncio.sleep(1.5)

    except asyncio.CancelledError:
        logger.info("Attack 2 (Botnet) execution stopped.")
    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_botnet_attack())
