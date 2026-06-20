from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from app.db.postgres import get_db
from app.db.models import ResponseAuditLog

router = APIRouter(prefix="/api/audit", tags=["Audit Logs"])


@router.get("")
async def get_all_audit_logs(
    limit: int = Query(100, le=500, description="Max number of records to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the most recent response audit log entries across all devices.
    Returns up to `limit` records (default 100, max 500), newest first.
    """
    result = await db.execute(
        select(ResponseAuditLog)
        .order_by(desc(ResponseAuditLog.timestamp))
        .limit(limit)
    )
    logs = result.scalars().all()
    return [_serialize(log) for log in logs]


@router.get("/{device_id}")
async def get_device_audit_logs(
    device_id: str,
    limit: int = Query(50, le=200, description="Max records to return for this device"),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the audit log history for a specific device.
    Shows all response actions taken (rate_limit, sandbox, quarantine, honeypot, release, recover).
    """
    result = await db.execute(
        select(ResponseAuditLog)
        .where(ResponseAuditLog.device_id == device_id)
        .order_by(desc(ResponseAuditLog.timestamp))
        .limit(limit)
    )
    logs = result.scalars().all()
    if not logs:
        raise HTTPException(
            status_code=404,
            detail=f"No audit records found for device '{device_id}'"
        )
    return [_serialize(log) for log in logs]


@router.get("/action/{action}")
async def get_audit_logs_by_action(
    action: str,
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Filter audit logs by action type.
    Valid values: rate_limit, sandbox, quarantine, honeypot, release, recover, denied.
    """
    result = await db.execute(
        select(ResponseAuditLog)
        .where(ResponseAuditLog.action == action)
        .order_by(desc(ResponseAuditLog.timestamp))
        .limit(limit)
    )
    logs = result.scalars().all()
    return [_serialize(log) for log in logs]


def _serialize(log: ResponseAuditLog) -> dict:
    """Convert a ResponseAuditLog ORM object to a JSON-serializable dict."""
    return {
        "id": log.id,
        "device_id": log.device_id,
        "trigger_score": round(log.trigger_score, 2),
        "response_tier": log.response_tier,
        "action": log.action,
        "hitl_decision": log.hitl_decision,
        "notes": log.notes,
        "shap_evidence": log.shap_evidence,
        "timestamp": log.timestamp.isoformat() + "Z",
    }
