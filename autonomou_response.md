# DeviceDNA Phase 2: Autonomous Response Engine — Implementation Plan

This plan bridges the gap between our current basic `ResponseEngine` (Phase 1/Hackathon Phase 7) and the comprehensive **Phase 2 PRD**.

## Current Status (What we have achieved)
- ✅ **Basic Response Engine**: We have `response_engine.py` using Redis to track isolated/sandboxed/forensic states via TTL keys.
- ✅ **Hardcoded Triggers**: We trigger actions based on simple score thresholds (e.g., `< 20` isolate, `< 40` sandbox, `> 30 point drop` forensic).
- ✅ **WebSocket Notifications**: Actions emit WebSocket events to the frontend.
- ✅ **Underlying ML Trust Engine**: The 5-pillar trust score evaluation cycle is fully robust.

## Gap Analysis (What we need for Phase 2)
The Phase 2 PRD introduces an enterprise-grade, 5-tier progressive escalation model, human-in-the-loop (HITL) overrides, and adaptive trust decay. 

## Proposed Implementation Roadmap

### 1. MQTT Command Dispatcher & ESP32 Integration
Currently, responses just set Redis keys. We need to physically actuate the hardware.
- **Backend (`mqtt_dispatcher.py`)**: Publish JSON command payloads (`{"action": "quarantine", "relay_open": true}`) to `devicedna/{device_id}/command`.
- **ESP32 Firmware**: Update firmware `messageReceived()` to parse commands, adjust MQTT publish delays (rate limiting), and trigger GPIO 26 (relay quarantine).

### 2. Risk Classifier & Adaptive Trust Decay
Replace the hardcoded thresholds in `evaluate_triggers()` with the 5-tier system.
- **Trust Decay (`trust_decay.py`)**: Track anomaly event timestamps in a 60-minute rolling Redis Sorted Set. Calculate the decay multiplier (`max(0.40, 1.0 - (event_count * 0.12))`).
- **Risk Tiers**: Map Effective Trust to Tier 1 (Monitor), Tier 2 (Rate Limit), Tier 3 (Sandbox), Tier 4 (Quarantine), and Tier 5 (Honeypot).

### 3. Audit Logging (PostgreSQL)
- **Schema**: Create a `ResponseAuditLog` SQLAlchemy model (event_id, device_id, trigger_score, response_tier, action, hitl_decision).
- **Integration**: Every time the response engine fires, write to the DB instead of just `logger.warning`.

### 4. Human-in-the-Loop (HITL) Override System
- **API Endpoints**: `GET /api/response/pending`, `POST /api/response/approve/{id}`, `POST /api/response/deny/{id}`.
- **Backend Queue**: Tier 4/5 actions enter a pending state in Redis with a 2-minute expiration.
- **Frontend Panel**: A modal in the React dashboard for pending approvals showing SHAP evidence and countdown timers.

### 5. Recovery Manager
- **Logic**: Track 5-minute "clean" windows (Trust > 70, no anomalies) in Redis.
- **Restoration**: Gradually restore trust (+5 per window for Tier 2, etc.) and auto-release devices from Sandbox/Rate Limit.

### 6. Peer Consensus & Immunization (Advanced)
- **Consensus**: Compare a flagged device's vector against peers of the same class. If unique, escalate confidence.
- **Immunization**: If `cam_01` hits Tier 4, dynamically broadcast `devicedna/broadcast/immunize` to tighten thresholds for all cameras.

## User Review Required
> [!IMPORTANT] 
> Do you want to implement the **Full ESP32 firmware updates** physically, or simulate the MQTT responses in the Python simulator first for rapid testing?

> [!NOTE]
> The Honeypot (Tier 5) is marked as OPTIONAL in the PRD. Should we prioritize Tiers 1-4 and HITL first?

Please review this roadmap. We can begin with Step 1: MQTT Command Dispatcher upon your approval!
