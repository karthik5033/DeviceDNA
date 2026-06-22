import asyncio
import json
import time
from aiokafka import AIOKafkaConsumer

async def main():
    consumer = AIOKafkaConsumer(
        'raw-flows',
        bootstrap_servers='localhost:29092',
        auto_offset_reset='latest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    await consumer.start()
    
    counts = {}
    print("Reading new messages from Kafka raw-flows topic for 5 seconds...")
    
    start_time = time.time()
    try:
        while time.time() - start_time < 5.0:
            try:
                # wait for a message with a short timeout
                msg = await asyncio.wait_for(consumer.getone(), timeout=0.5)
                flow = msg.value
                dev_id = flow.get('device_id', 'unknown')
                counts[dev_id] = counts.get(dev_id, 0) + 1
            except asyncio.TimeoutError:
                continue
    finally:
        await consumer.stop()

    print("\nNew flow count per device ID in raw-flows topic:")
    for dev_id, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dev_id}: {count}")

if __name__ == "__main__":
    asyncio.run(main())
