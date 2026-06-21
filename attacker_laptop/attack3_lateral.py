import asyncio
import logging
import random
from attacker_laptop.common.config import FLEET, TOPIC_NAME
from attacker_laptop.common.producer import AttackKafkaProducer
from attacker_laptop.common.attack_utils import create_raw_flow

logger = logging.getLogger(__name__)

async def run_lateral_attack(target_id: str = "cam_01"):
    """
    Attack 3 - Lateral Movement
    Behavior: Spawns topological connections internally between target and devices of different classes.
    Expected Detector: GraphSAGE GNN
    Expected Trust: drops from ~95 to 20.
    """
    device = next((d for d in FLEET if d["id"] == target_id), None)
    if not device:
        logger.error(f"Target device {target_id} not found in fleet configuration.")
        return

    logger.info(f"🚀 Starting Attack 3 (Lateral Movement) targeting device: {target_id} ({device['ip_address']})")
    
    producer = AttackKafkaProducer()
    await producer.start()

    try:
        # Find internal targets of other classes (like medical, thermostat, sensor)
        targets = [
            d for d in FLEET 
            if d["device_class"] in ["medical", "thermostat", "sensor"] and d["id"] != target_id
        ]
        
        if not targets:
            logger.error("No suitable internal peer targets found for lateral movement.")
            return

        while True:
            # Pick a target internally
            peer_target = random.choice(targets)
            
            # Lateral profile: TCP connection internally (e.g. SSH port 22 or Modbus port 502)
            dst_port = 22 if peer_target["device_class"] == "medical" else 502
            flow = create_raw_flow(
                device_id=device["id"],
                device_class=device["device_class"],
                src_ip=device["ip_address"],
                dst_ip=peer_target["ip_address"],
                dst_port=dst_port,
                protocol="TCP",
                bytes_count=2048,
                packets_count=15,
                flags="TCP_SYN",
                is_anomalous=True,
                attack_type="lateral_movement"
            )
            
            await producer.send_flow(TOPIC_NAME, flow)
            logger.info(f"Sent Lateral flow: {device['ip_address']} -> {peer_target['ip_address']}:{dst_port} (Internal Edge)")
            
            # Send periodic lateral attempts every 2 seconds
            await asyncio.sleep(2.0)

    except asyncio.CancelledError:
        logger.info("Attack 3 (Lateral) execution stopped.")
    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_lateral_attack())
