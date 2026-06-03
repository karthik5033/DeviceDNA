import asyncio
import os
import sys
from datetime import datetime

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import select
from app.db.postgres import AsyncSessionLocal, engine
from app.db.models import Alert, ResponseAuditLog

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ".center(80, "="))
    print("=" * 80)

def print_row(cols: list, widths: list):
    row_str = " | ".join(f"{str(col)[:w].ljust(w)}" for col, w in zip(cols, widths))
    print(f"| {row_str} |")

def print_divider(widths: list):
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")

async def view_alerts():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Alert).order_by(Alert.timestamp.desc()).limit(10)
        )
        alerts = result.scalars().all()
        
        print_header("Latest 10 Security Alerts (PostgreSQL)")
        if not alerts:
            print("No alerts found in database.")
            return
            
        widths = [36, 12, 10, 20, 10, 20]
        headers = ["Alert ID", "Device ID", "Severity", "Alert Type", "Trust Score", "Timestamp (UTC)"]
        
        print_divider(widths)
        print_row(headers, widths)
        print_divider(widths)
        
        for a in alerts:
            ts_str = a.timestamp.strftime("%Y-%m-%d %H:%M:%S") if a.timestamp else "N/A"
            print_row([
                a.id, 
                a.device_id, 
                a.severity.upper(), 
                a.alert_type, 
                f"{a.trust_score:.2f}", 
                ts_str
            ], widths)
            
        print_divider(widths)

async def view_audit_logs():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ResponseAuditLog).order_by(ResponseAuditLog.timestamp.desc()).limit(10)
        )
        logs = result.scalars().all()
        
        print_header("Latest 10 Autonomous Response Actions (PostgreSQL)")
        if not logs:
            print("No response audit logs found in database.")
            return
            
        widths = [36, 12, 12, 10, 15, 20]
        headers = ["Log ID", "Device ID", "Trigger Score", "Tier", "Action", "Timestamp (UTC)"]
        
        print_divider(widths)
        print_row(headers, widths)
        print_divider(widths)
        
        for l in logs:
            ts_str = l.timestamp.strftime("%Y-%m-%d %H:%M:%S") if l.timestamp else "N/A"
            print_row([
                l.id, 
                l.device_id, 
                f"{l.trigger_score:.2f}", 
                f"Tier {l.response_tier}", 
                l.action, 
                ts_str
            ], widths)
            
        print_divider(widths)

async def main():
    try:
        await view_alerts()
        await view_audit_logs()
    except Exception as e:
        print(f"Error querying PostgreSQL database: {e}", file=sys.stderr)
    finally:
        await engine.dispose()

if __name__ == "__main__":
    # If running on Windows, set correct selector event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
