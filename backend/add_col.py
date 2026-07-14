import asyncio
from app.db.postgres import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text("ALTER TABLE response_audit_logs ADD COLUMN shap_evidence JSON;"))
            await session.commit()
            print("Column 'shap_evidence' added successfully.")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
