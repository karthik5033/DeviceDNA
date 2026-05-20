from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.postgres import get_db
from app.db.models import Alert

router = APIRouter()

@router.get("/alerts")
async def get_alerts(db: AsyncSession = Depends(get_db)):
    """Retrieve the last 50 alerts ordered by timestamp descending."""
    result = await db.execute(select(Alert).order_by(Alert.timestamp.desc()).limit(50))
    alerts = result.scalars().all()
    return alerts

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    """Mark an alert as resolved."""
    result = await db.execute(select(Alert).filter(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    alert.is_resolved = True
    await db.commit()
    await db.refresh(alert)
    return alert
