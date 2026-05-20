import asyncio
from app.db.postgres import engine, Base
from app.db.models import Alert

async def drop_and_create():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("Recreated tables")

asyncio.run(drop_and_create())
