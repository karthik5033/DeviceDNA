import json
import logging
from kafka import KafkaProducer
from attacker_laptop.common.config import KAFKA_BROKER

logger = logging.getLogger(__name__)

class AttackKafkaProducer:
    def __init__(self, kafka_broker: str = None):
        self.producer = None
        self.kafka_broker = kafka_broker or KAFKA_BROKER

    async def start(self):
        logger.info(f"Connecting attacker producer to Kafka broker at {self.kafka_broker}...")
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info("Attacker producer successfully connected to Kafka.")

    async def send_flow(self, topic: str, flow: dict):
        if self.producer and flow:
            # send() is asynchronous and buffers flows internally; flush forces send
            self.producer.send(topic, value=flow)
            self.producer.flush()

    async def stop(self):
        if self.producer:
            self.producer.close()
            logger.info("Attacker producer stopped.")


