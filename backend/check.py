import asyncio
from app.db.postgres import AsyncSessionLocal
from sqlalchemy import select
from app.db.models import ResponseAuditLog

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ResponseAuditLog).limit(1))
        log = result.scalars().first()
        if log:
            print(repr(log.timestamp), type(log.timestamp))
            if getattr(log.timestamp, 'tzinfo', None):
                print('HAS TZINFO')
            else:
                print('NO TZINFO')
        else:
            print('No logs found')

if __name__ == "__main__":
    asyncio.run(main())
