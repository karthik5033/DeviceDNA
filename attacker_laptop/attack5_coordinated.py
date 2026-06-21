import asyncio
import logging
import random
from attacker_laptop.common.config import FLEET, TOPIC_NAME
from attacker_laptop.common.producer import AttackKafkaProducer
from attacker_laptop.common.attack_utils import create_raw_flow

logger = logging.getLogger(__name__)

async def run_coordinated_attack():
    """
    Attack 5 - Coordinated Multi-device Compromise
    Behavior: Orchestrates simultaneous C2 beaconing and peer-to-peer lateral connections 
              among SIM-0001, SIM-0007, SIM-0021, and SIM-0032.
    Expected Detector: VAE, Isolation Forest, LSTM, GNN, CUSUM
    Expected Trust: Simultaneous drops and cascade alerts.
    """
    compromised_ids = ["SIM-0001", "SIM-0007", "SIM-0021", "SIM-0032"]
    compromised_devices = [d for d in FLEET if d["id"] in compromised_ids]
    
    if len(compromised_devices) < len(compromised_ids):
        logger.warning("Some targeted devices (SIM-0001, SIM-0007, SIM-0021, SIM-0032) could not be found.")
        # Proceed with whatever target devices we found
        if not compromised_devices:
            logger.error("No target devices found.")
            return

    logger.info(f"🚀 Starting Attack 5 (Coordinated Multi-device) targeting: {[d['id'] for d in compromised_devices]}")
    
    producer = AttackKafkaProducer()
    await producer.start()

    try:
        c2_ip = "198.51.100.99"
        
        while True:
            # 1. Simulate external beaconing for all compromised nodes
            for dev in compromised_devices:
                beacon_flow = create_raw_flow(
                    device_id=dev["id"],
                    device_class=dev["device_class"],
                    src_ip=dev["ip_address"],
                    dst_ip=c2_ip,
                    dst_port=443,
                    protocol="HTTPS",
                    bytes_count=256,
                    packets_count=4,
                    flags="TCP_ACK",
                    is_anomalous=True,
                    attack_type="botnet_c2"
                )
                await producer.send_flow(TOPIC_NAME, beacon_flow)
                logger.info(f"Coordinated Beacon: {dev['id']} ({dev['ip_address']}) -> C2 ({c2_ip})")
            
            # 2. Simulate internal lateral communication among the compromised group (forming a cluster)
            if len(compromised_devices) > 1:
                for src in compromised_devices:
                    # Pick a different compromised peer
                    dst = random.choice([d for d in compromised_devices if d["id"] != src["id"]])
                    lateral_flow = create_raw_flow(
                        device_id=src["id"],
                        device_class=src["device_class"],
                        src_ip=src["ip_address"],
                        dst_ip=dst["ip_address"],
                        dst_port=8080, # Custom port
                        protocol="TCP",
                        bytes_count=1024,
                        packets_count=10,
                        flags="TCP_SYN",
                        is_anomalous=True,
                        attack_type="lateral_movement"
                    )
                    await producer.send_flow(TOPIC_NAME, lateral_flow)
                    logger.info(f"Coordinated Peer Lateral: {src['id']} -> {dst['id']} (Internal Edge)")
            
            # Run coordinated steps every 3 seconds
            await asyncio.sleep(3.0)

    except asyncio.CancelledError:
        logger.info("Attack 5 (Coordinated) execution stopped.")
    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_coordinated_attack())
