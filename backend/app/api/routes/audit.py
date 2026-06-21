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


@router.get("/summary")
async def get_audit_summary(db: AsyncSession = Depends(get_db)):
    """
    Returns a high-level summary of all response actions taken.
    Includes: total count, breakdown by action, breakdown by tier, breakdown by HITL decision.
    """
    from sqlalchemy import func, case

    total_res = await db.execute(select(func.count()).select_from(ResponseAuditLog))
    total = total_res.scalar()

    # By action
    action_res = await db.execute(
        select(ResponseAuditLog.action, func.count().label("count"))
        .group_by(ResponseAuditLog.action)
        .order_by(func.count().desc())
    )
    by_action = {row.action: row.count for row in action_res.all()}

    # By tier
    tier_res = await db.execute(
        select(ResponseAuditLog.response_tier, func.count().label("count"))
        .group_by(ResponseAuditLog.response_tier)
        .order_by(ResponseAuditLog.response_tier)
    )
    by_tier = {f"tier_{row.response_tier}": row.count for row in tier_res.all()}

    # By HITL decision
    hitl_res = await db.execute(
        select(ResponseAuditLog.hitl_decision, func.count().label("count"))
        .group_by(ResponseAuditLog.hitl_decision)
        .order_by(func.count().desc())
    )
    by_decision = {row.hitl_decision: row.count for row in hitl_res.all()}

    # Most actioned device
    top_device_res = await db.execute(
        select(ResponseAuditLog.device_id, func.count().label("count"))
        .group_by(ResponseAuditLog.device_id)
        .order_by(func.count().desc())
        .limit(5)
    )
    top_devices = [{"device_id": row.device_id, "count": row.count} for row in top_device_res.all()]

    return {
        "total_actions": total,
        "by_action": by_action,
        "by_tier": by_tier,
        "by_hitl_decision": by_decision,
        "top_actioned_devices": top_devices,
    }

