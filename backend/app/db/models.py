import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, Integer
from app.db.postgres import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String, index=True, nullable=False)
    severity = Column(String, nullable=False)  # critical, high, medium
    alert_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    
    trust_score = Column(Float, nullable=False)
    vae_score = Column(Float, nullable=False)
    if_score = Column(Float, nullable=False)
    lstm_score = Column(Float, nullable=False)
    gnn_score = Column(Float, nullable=False)
    tib = Column(JSON, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    is_resolved = Column(Boolean, default=False)

class ResponseAuditLog(Base):
    __tablename__ = "response_audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String, index=True, nullable=False)
    trigger_score = Column(Float, nullable=False)
    response_tier = Column(Integer, nullable=False)
    action = Column(String, nullable=False)
    hitl_decision = Column(String, nullable=False) # approved, denied, automatic
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
