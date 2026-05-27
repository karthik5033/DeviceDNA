from fastapi import APIRouter
from pydantic import BaseModel
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
