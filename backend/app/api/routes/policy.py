from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.postgres import get_db
from app.db.models import PolicyRule
from app.ml.nlp.policy_parser import parse_policy

router = APIRouter(prefix="/policy", tags=["Policy"])

class PolicyParseRequest(BaseModel):
    statement: str

@router.post("/parse")
async def parse_policy_endpoint(payload: PolicyParseRequest):
    """
    Parse a plain-English security policy statement into a structured rule.
    """
    result = parse_policy(payload.statement)
    return result

@router.post("/create")
async def create_policy_rule(payload: PolicyParseRequest, db: AsyncSession = Depends(get_db)):
    """
    Parse a plain-English statement and save it as an active PolicyRule in the database.
    """
    parsed = parse_policy(payload.statement)
    if parsed.get("condition") == "unknown_condition":
        raise HTTPException(status_code=400, detail="Could not parse statement into a valid policy rule condition.")

    rule = PolicyRule(
        device_class=parsed.get("device_class", "any"),
        condition=parsed.get("condition"),
        time_constraint=parsed.get("time_constraint"),
        action=parsed.get("action", "alert"),
        severity=parsed.get("severity", "MEDIUM"),
        natural_language_rule=parsed.get("natural_language_rule"),
        parse_confidence=parsed.get("parse_confidence", 1.0),
        is_active=True
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return rule

@router.get("/list")
async def list_policy_rules(db: AsyncSession = Depends(get_db)):
    """
    Retrieve all policy rules.
    """
    result = await db.execute(select(PolicyRule).order_by(PolicyRule.timestamp.desc()))
    rules = result.scalars().all()
    return rules

@router.post("/toggle/{rule_id}")
async def toggle_policy_rule(rule_id: str, db: AsyncSession = Depends(get_db)):
    """
    Toggle the active status of a policy rule.
    """
    result = await db.execute(select(PolicyRule).filter(PolicyRule.id == rule_id))
    rule = result.scalar_one_or_none()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Policy rule not found")
        
    rule.is_active = not rule.is_active
    await db.commit()
    await db.refresh(rule)
    return rule

@router.delete("/{rule_id}")
async def delete_policy_rule(rule_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a policy rule.
    """
    result = await db.execute(select(PolicyRule).filter(PolicyRule.id == rule_id))
    rule = result.scalar_one_or_none()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Policy rule not found")
        
    await db.delete(rule)
    await db.commit()
    return {"status": "success", "message": f"Policy rule {rule_id} deleted."}

