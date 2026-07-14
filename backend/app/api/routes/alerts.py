from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.db.postgres import get_db
from app.db.models import Alert

router = APIRouter()

def _serialize_alert(alert: Alert) -> dict:
    return {
        "id": str(alert.id),
        "device_id": alert.device_id,
        "severity": alert.severity,
        "alert_type": alert.alert_type,
        "message": alert.message,
        "trust_score": alert.trust_score,
        "vae_score": alert.vae_score,
        "if_score": alert.if_score,
        "lstm_score": alert.lstm_score,
        "gnn_score": alert.gnn_score,
        "tib": alert.tib,
        "is_resolved": alert.is_resolved,
        "timestamp": alert.timestamp.isoformat().replace("+00:00", "Z") if getattr(alert.timestamp, "tzinfo", None) else (alert.timestamp.isoformat() + "Z" if alert.timestamp else None),
    }

@router.get("/alerts")
async def get_alerts(db: AsyncSession = Depends(get_db)):
    """Retrieve the last 50 alerts ordered by timestamp descending."""
    result = await db.execute(select(Alert).order_by(Alert.timestamp.desc()).limit(50))
    alerts = result.scalars().all()
    return [_serialize_alert(a) for a in alerts]

@router.get("/alerts/count/resolved")
async def get_resolved_alerts_count(db: AsyncSession = Depends(get_db)):
    """Return total count of resolved alerts (used by dashboard KPI)."""
    result = await db.execute(
        select(func.count()).where(Alert.is_resolved == True)
    )
    count = result.scalar() or 0
    return {"count": count}

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
    return _serialize_alert(alert)
