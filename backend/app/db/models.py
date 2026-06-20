import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, Integer, Text
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
    hitl_decision = Column(String, nullable=False)  # approved, denied, automatic, manual_override
    notes = Column(Text, nullable=True)              # Human-readable context note
    shap_evidence = Column(JSON, nullable=True)      # SHAP feature attribution dict
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class PolicyRule(Base):
    __tablename__ = "policy_rules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_class = Column(String, nullable=False, default="any") # e.g. camera, printer, sensor, any
    condition = Column(String, nullable=False) # e.g. ext_int_ratio > 0.8
    time_constraint = Column(String, nullable=True) # e.g. hour >= 0 AND hour < 6
    action = Column(String, nullable=False, default="alert") # alert, isolate
    severity = Column(String, nullable=False, default="MEDIUM") # LOW, MEDIUM, HIGH, CRITICAL
    natural_language_rule = Column(String, nullable=True)
    parse_confidence = Column(Float, nullable=True, default=1.0)
    is_active = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
