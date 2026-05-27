import asyncio
import os
import sys

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.influxdb import InfluxDBService

async def main():
    try:
        influx = InfluxDBService()
        res = await influx.query_trust_history('SIM-0030', hours=6)
        print("COUNT:", len(res))
        await influx.close()
    except Exception as e:
        print("EXCEPTION:", e)

if __name__ == '__main__':
    asyncio.run(main())
