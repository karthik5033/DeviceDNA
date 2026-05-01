# DeviceDNA — Project Status Report

> **Last Updated**: April 29, 2026  
> **Build Plan Reference**: `plan/00_master_build_flow.md`

---

## Overall Summary

DeviceDNA has a **solid foundation** across all layers — the Docker infrastructure, telemetry simulator, 4 ML model architectures, a Kafka-based data pipeline, and a polished Next.js SOC dashboard are all scaffolded and partially operational. However, many subsystems currently run on **mock/hardcoded data** rather than live integrations, and several key ML models exist as code-only (not yet integrated into the live scoring pipeline).

```
Phase 0  ████████████████████  COMPLETE         — Infrastructure & Scaffolding
Phase 1  ██████████████░░░░░░  ~75% DONE        — Simulator & Data Pipeline
Phase 2  █████████░░░░░░░░░░░  ~50% DONE        — Digital Twin & DNA Fingerprint
Phase 3  ████████░░░░░░░░░░░░  ~40% DONE        — ML Ensemble & Trust Engine
Phase 4  ░░░░░░░░░░░░░░░░░░░░  NOT STARTED      — Explainability (SHAP)
Phase 5  ██████████████████░░  ~90% DONE        — SOC Dashboard Frontend
Phase 6  █████░░░░░░░░░░░░░░░  ~25% DONE (UI)   — NLP Policy & Advanced Features
Phase 7  █░░░░░░░░░░░░░░░░░░░  ~5% DONE         — Autonomous Response
Phase 8  ░░░░░░░░░░░░░░░░░░░░  NOT STARTED      — Polish & Deployment
```

---

## Phase 0 — Infrastructure & Scaffolding ✅ COMPLETE

Everything in Phase 0 is done and working.

| Task | Status | Notes |
|------|--------|-------|
| Next.js 14 Frontend | ✅ Done | App Router, TypeScript, Tailwind CSS |
| FastAPI Backend | ✅ Done | Async, Socket.IO ASGI wrapping, Uvicorn |
| Docker Compose | ✅ Done | PostgreSQL 16, InfluxDB 2.7, Redis 7, Kafka + Zookeeper |
| Database Schema (SQL) | ✅ Done | `devices`, `alerts`, `platform_settings` tables in `backend/db_init/init.sql` |
| Health Check Endpoint | ✅ Done | `GET /api/health` returns `{"status": "ok"}` |
| Environment Config | ✅ Done | `.env` with all DB/Kafka/Redis/Frontend URLs |
| CORS Setup | ✅ Done | `allow_origins=["*"]` in FastAPI middleware |

---

## Phase 1 — Telemetry Simulator & Data Pipeline ⚠️ ~75%

The simulator itself is fully functional. The Kafka pipeline works. But the **InfluxDB write-through** and **feature extraction loop** are not wired into the live consumer.

| Task | Status | Notes |
|------|--------|-------|
| 6 Device Class Profiles | ✅ Done | Camera, Sensor, Thermostat, Access Control, Medical, Industrial |
| 50-Device Fleet Generation | ✅ Done | Unique IDs (`SIM-0001` to `SIM-0050`), MAC, IP, VLAN |
| Normal Traffic Generator | ✅ Done | Per-class behavioral profiles (protocols, ports, byte sizes) |
| 4 Attack Scenarios | ✅ Done | C2 Botnet, Slow Exfil, Lateral Movement, NLP Policy Trigger |
| Kafka Producer (Simulator) | ✅ Done | Publishes to `raw-flows` topic, 100 flows/batch, injects attacks every 100 cycles |
| Kafka Consumer (Backend) | ✅ Done | `TelemetryService` listens to `raw-flows`, broadcasts via Socket.IO |
| Feature Extraction Engine | ✅ Code Done | `feature_extraction.py` computes 14-dim vectors, but... |
| InfluxDB Write-Through | ❌ **NOT WIRED** | `TelemetryService._process_flow()` does NOT call feature extraction or write to InfluxDB |
| 5-Min Rolling Aggregation | ❌ **MISSING** | No windowing logic exists — flows are processed one-by-one |

> **⚠️ Key Gap**: The Kafka consumer receives flows and broadcasts alerts via WebSocket, but it **never calls feature extraction**, **never writes to InfluxDB**, and **never triggers the ML scoring pipeline**. This is the biggest backend integration gap.

---

## Phase 2 — Digital Twin & DNA Fingerprinting ⚠️ ~50%

The VAE model architecture, training script, and scoring logic are complete. DNA fingerprinting service exists. But nothing is trained yet (no `models_trained/` directory with `.pt` files).

| Task | Status | Notes |
|------|--------|-------|
| VAE Architecture | ✅ Done | `DeviceVAE`: 14→32→16 latent dims with KLD+MSE loss |
| VAE Training Script | ✅ Done | `train_vae.py`: Generates synthetic baselines, trains per-device, saves `.pt` files |
| VAE Scoring Service | ✅ Done | `VAE_TwinScorer`: Loads twins, computes deviation score (0–1) |
| DNA Fingerprint Service | ✅ Done | `DNAFingerprintService`: 30-dim DNA vector, cosine similarity, unknown device classification |
| **Model Training Execution** | ❌ **NOT RUN** | No `models_trained/` directory exists — `python -m training.train_vae` has never been run |
| Baseline Data Collection | ❌ **NOT DONE** | Training script uses synthetic data (which is fine), but hasn't been executed |

> **ℹ️ Important**: Running `python -m training.train_vae` from the `backend/` directory will train all 50 Digital Twins. This is a prerequisite for the VAE scoring to actually work (currently returns -1.0 because no `.pt` files exist).

---

## Phase 3 — ML Detection Ensemble & Trust Engine ⚠️ ~40%

The 5-pillar Trust Score Engine is architected. VAE + Isolation Forest + CUSUM are code-complete. LSTM and GNN exist as model architectures only (not integrated into scoring).

| Task | Status | Notes |
|------|--------|-------|
| Isolation Forest Model | ✅ Code Done | `IF_AnomalyScorer`: Loads per-class `.joblib` models, normalizes scores |
| Isolation Forest Training | ❌ **MISSING** | No `train_isolation_forest.py` script exists in `training/` |
| LSTM Model Architecture | ✅ Code Done | `TimeSeriesLSTM`: 2-layer LSTM (14→64→14), sliding window prediction |
| LSTM Training Script | ❌ **MISSING** | No `train_lstm.py` in `training/` |
| LSTM Integration in Trust Engine | ❌ **STUBBED** | Trust engine hardcodes LSTM score as `0.0` (see `trust_engine.py` line 47) |
| GNN (GraphSAGE) Architecture | ✅ Code Done | `GraphSAGENetwork`: 2 SAGEConv layers, binary classification |
| GNN Training Script | ❌ **MISSING** | No `train_gnn.py` in `training/` |
| GNN Integration in Trust Engine | ❌ **STUBBED** | Trust engine hardcodes GNN score as `0.0` (see `trust_engine.py` line 47) |
| PyTorch Geometric Dependency | ⚠️ **Not Installed** | `torch-geometric` is NOT in `requirements.txt` — GNN model has a try/except fallback |
| CUSUM Drift Engine | ✅ Done | `CUSUMDriftEngine`: Stateful Z-score tracking with configurable slack/threshold |
| 5-Pillar Trust Score Engine | ✅ Architecture Done | `TrustScoreEngine`: Weighted pillars (VAE 35%, Ensemble 25%, Drift 20%, Policy 15%, Peer 5%) |
| Trust Score API Endpoint | ✅ Done | `POST /api/trust/evaluate` and `GET /api/trust/{device_id}/current` |
| Trust Score → Redis Cache | ❌ **NOT DONE** | Noted as "Missing feature" in `trust_engine.py` line 91 |
| Trust Score → InfluxDB History | ❌ **NOT DONE** | Scores are computed but never persisted |
| Policy Conformance Pillar | ❌ **STUBBED** | Hardcoded to `0.0` — no policy evaluation engine exists |
| Peer Comparison Pillar | ❌ **STUBBED** | Hardcoded to `0.0` — DNA comparison not wired in |
| `GET /api/trust/{id}/current` | ⚠️ **MOCK** | Always returns `trust_score: 100.0` — doesn't query Redis/DB |

---

## Phase 4 — Explainability Engine ❌ NOT STARTED

| Task | Status | Notes |
|------|--------|-------|
| SHAP Integration | ❌ Not Started | No `shap` in `requirements.txt`, no explainability service |
| Feature-to-Language Mapping | ❌ Not Started | The 120-entry lookup table doesn't exist |
| Threat Intelligence Brief (TIB) Generator | ❌ Not Started | |
| Alert Auto-Creation Pipeline | ❌ Not Started | Alerts are mock/hardcoded in the frontend |
| Alert API Endpoints | ❌ Not Started | No `alerts.py` route file in backend API |

---

## Phase 5 — SOC Dashboard Frontend ✅ ~90%

The frontend is the most complete layer. It's polished, animated, and functional for demo purposes.

| Task | Status | Notes |
|------|--------|-------|
| Landing Page | ✅ Done | Premium glassmorphism design, animated background, API status indicator |
| Dashboard Layout | ✅ Done | Sidebar + Header + scrollable content area |
| SOC Overview Page | ✅ Done | KPI cards, live D3 network map, alert queue, Recharts timeline |
| Network Topology Map (D3) | ✅ Done | 50-node force-directed graph, trust-score coloring, glow on anomaly, drag, click-to-drill |
| Trust Score Timeline (Recharts) | ✅ Done | 12-hour mock timeseries, live updates on WebSocket alerts, auto-recovery, threshold line |
| CUSUM Drift Heatmap | ✅ Done | 7-day × 24-hour calendar grid with color-coded drift scores, custom tooltips |
| Alerts Page | ✅ Done | Full alert feed, severity filtering, SHAP context panel (mock), device isolation button |
| Topology Full Page | ✅ Done | Expanded D3 map with legend, filter/export buttons (UI only) |
| Predictive Risk Page | ✅ Done | CUSUM stats cards, drift heatmap, LSTM forecast panel (descriptive, not live) |
| NLP Policy Page | ✅ Done | Text input, simulated BERT translation with fake results, active policy ledger |
| Attack Replay Page | ✅ Done | Timeline scrub, transport controls (play/pause/seek), forensic flow capture panel |
| WebSocket Integration | ✅ Done | `socket.io-client` connects to backend, handles `new_alert`, `telemetry_ping`, `device_isolated` |
| Device Detail Drawer | ❌ **MISSING** | No dedicated device drill-down panel/page |
| Zustand State Store | ⚠️ Empty | `frontend/store/` exists but is empty — all state is local `useState` |
| API Client (`lib/api.ts`) | ❌ **MISSING** | No centralized API client — only the health check fetch exists |
| Loading States / Skeletons | ❌ **MISSING** | No skeleton screens or loading indicators |

> **ℹ️ Note**: All dashboard data is currently **mock/simulated on the frontend**. The D3 network map generates its own fake nodes. Trust timeline generates its own fake history. Alerts are hardcoded. The real backend APIs are not being called for data.

---

## Phase 6 — NLP Policy Engine & Advanced Features ⚠️ ~25% (UI Only)

| Task | Status | Notes |
|------|--------|-------|
| NLP Policy UI | ✅ Done | Beautiful textarea + translation result panel |
| BERT Policy Parser | ❌ Not Started | No `nlp/` ML model directory, no training script, no HuggingFace dependency |
| Intent Classification | ❌ Not Started | Frontend simulates this with `setTimeout` |
| Entity Extraction | ❌ Not Started | |
| Rule Generation Engine | ❌ Not Started | |
| Policy Evaluation Integration | ❌ Not Started | |
| Attack Replay (Backend) | ❌ Not Started | Replay page exists but uses same live timeline mock data |
| What-If Simulator | ❌ Not Started | Not even a page for this yet |

---

## Phase 7 — Autonomous Response ⚠️ ~5%

| Task | Status | Notes |
|------|--------|-------|
| Device Isolation (WebSocket) | ✅ Done | `isolate_device` WebSocket event handler with simulated delay |
| Response Action Library | ❌ Not Started | Only "isolate" exists, no sandbox/throttle/quarantine/block |
| Autonomous Response Rules | ❌ Not Started | |
| Response Mode Toggle | ⚠️ DB Seeded | `platform_settings` has `response_mode: "advisory"` but no code reads it |
| Response Audit Log | ❌ Not Started | |
| Honey-Patch Sandbox | ❌ Not Started | |

---

## Phase 8 — Polish & Deployment ❌ NOT STARTED

| Task | Status |
|------|--------|
| Demo Script | ❌ |
| Performance Optimization | ❌ |
| Error Handling / Resilience | ❌ |
| Docker Compose Finalization (backend included) | ❌ |
| Pre-loaded Demo Data | ❌ |
| Full Documentation | ⚠️ Partial (README exists) |

---

## 🔴 Critical Issues to Fix

These are blockers preventing the platform from functioning as intended:

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **Telemetry pipeline doesn't call feature extraction or ML scoring** | `backend/app/services/telemetry.py` `_process_flow()` | The entire ML pipeline is disconnected from live data |
| 2 | **No trained models exist** (`models_trained/` missing) | `backend/training/train_vae.py` | VAE scoring returns -1.0, IF scoring returns 0.0 |
| 3 | **`GET /api/trust/{id}/current` is hardcoded** to 100.0 | `backend/app/api/routes/trust.py` line 40 | Dashboard can never show real scores |
| 4 | **Frontend uses all mock data** — never calls backend APIs for real state | All dashboard pages | Dashboard looks alive but is entirely self-generated |
| 5 | **Trust scores not persisted** to Redis or InfluxDB | `backend/app/services/trust_engine.py` line 91 | No historical trust data, no cache layer |
| 6 | **PyTorch, scikit-learn, numpy NOT in requirements.txt** | `backend/requirements.txt` | ML code can't run without these dependencies |
| 7 | **`torch-geometric` not installed** | `backend/app/ml/gnn/model.py` | GNN model fails to import |

---

## 🟡 Things That Need To Be Done Next (Priority Order)

### Tier 1 — Make the ML Pipeline Actually Work

1. **Add missing Python dependencies** to `requirements.txt`:
   - `torch`, `torchvision`, `numpy`, `scikit-learn`, `joblib`
   - Optionally: `torch-geometric` (for GNN)
   - Optionally: `shap` (for explainability)

2. **Run VAE training** — Execute `python -m training.train_vae` from `backend/` to generate the 50 `.pt` Digital Twin models.

3. **Create Isolation Forest training script** (`training/train_isolation_forest.py`) — Train per-class IF models and save as `.joblib`.

4. **Wire the telemetry pipeline end-to-end**:
   - Add 5-minute rolling window aggregation in `TelemetryService`
   - Call `feature_extraction.extract_features()` on each window
   - Feed features through `master_trust_engine.evaluate_device()`
   - Persist scores to Redis (live cache) and InfluxDB (history)
   - Broadcast updated trust scores via WebSocket

5. **Make `GET /api/trust/{id}/current` read from Redis** instead of returning hardcoded 100.0.

### Tier 2 — Connect Frontend to Real Backend

6. **Create a centralized API client** (`frontend/lib/api.ts`) that fetches real data from the FastAPI backend.

7. **Replace mock data in dashboard pages** with API calls:
   - D3 Network Topology → fetch real device list + trust scores
   - Trust Timeline → fetch from InfluxDB history API
   - Alerts → fetch from backend alert API
   - Drift Heatmap → fetch from CUSUM engine API

8. **Add Zustand stores** for global state management (device list, alerts, trust scores).

### Tier 3 — Complete the ML Ensemble

9. **Create LSTM training script** and integrate temporal scoring into the trust engine.
10. **Create GNN training script** and integrate topological scoring (requires `torch-geometric`).
11. **Implement LSTM and GNN scoring services** (similar to `vae/scoring.py` and `isolation_forest/model.py`).
12. **Replace the hardcoded 0.0 values** in `trust_engine.py` line 47 with actual LSTM/GNN scores.

### Tier 4 — Explainability & Policy Engine

13. **Add SHAP explainability** — Compute SHAP values for IF/VAE outputs.
14. **Build Threat Intelligence Brief (TIB) generator**.
15. **Create alert auto-generation pipeline** — Trust score drop → alert creation → PostgreSQL persistence.
16. **Implement NLP policy parser backend** (BERT fine-tuning or rule-based fallback).

### Tier 5 — Advanced Features

17. **Device detail drawer/page** on the frontend.
18. **Attack replay backend** — Query historical InfluxDB data by time range.
19. **What-if simulator** — Simulate trust score changes if a device is isolated.
20. **Autonomous response engine** — Trust threshold → auto-isolate/sandbox/throttle.
21. **Loading states and error boundaries** on the frontend.

---

## 📁 Files That Don't Exist Yet (Per Master Plan)

| Planned File | Purpose |
|-------------|---------|
| `backend/app/config.py` | Centralized settings & env management |
| `backend/app/db/postgres.py` | PostgreSQL connection pool |
| `backend/app/db/redis.py` | Redis client |
| `backend/app/services/policy_engine.py` | NLP policy evaluation |
| `backend/app/services/explainability.py` | SHAP + TIB generation |
| `backend/app/services/response_engine.py` | Autonomous response actions |
| `backend/app/services/sandbox.py` | Honey-patch sandbox |
| `backend/app/ml/nlp/` | BERT policy parser model |
| `backend/app/ml/ensemble.py` | Weighted ensemble orchestration |
| `backend/app/api/routes/devices.py` | Device CRUD endpoints |
| `backend/app/api/routes/alerts.py` | Alert endpoints |
| `backend/app/api/routes/policies.py` | Policy CRUD + NLP endpoints |
| `backend/app/api/routes/response.py` | Response action triggers |
| `backend/app/api/routes/replay.py` | Historical replay data |
| `backend/app/api/routes/simulator.py` | What-if simulation |
| `backend/training/train_isolation_forest.py` | IF training script |
| `backend/training/train_lstm.py` | LSTM training script |
| `backend/training/train_gnn.py` | GNN training script |
| `backend/training/train_nlp.py` | NLP fine-tuning script |
| `frontend/lib/api.ts` | Backend API client |
| `frontend/lib/websocket.ts` | Centralized Socket.IO client |
| `frontend/lib/store.ts` | Zustand stores |
| `frontend/hooks/` | Custom React hooks |
| `frontend/types/` | TypeScript type definitions |
| `frontend/app/dashboard/devices/` | Device list + detail page |
| `frontend/app/dashboard/simulator/` | What-if simulator page |

---

## Quick Reference — What Works Right Now

| Component | Works? | Notes |
|-----------|--------|-------|
| `docker compose up` | ✅ | Starts Postgres, InfluxDB, Redis, Kafka |
| `uvicorn app.main:app --reload` | ✅ | FastAPI starts, health check works, WebSocket ready |
| `python -m simulator.main` | ✅ | Streams flows to Kafka, injects attacks |
| `npm run dev` (frontend) | ✅ | Landing page + full dashboard render with mock data |
| Backend → Frontend WebSocket | ✅ | Live alerts stream from Kafka through to the dashboard |
| `POST /api/trust/evaluate` | ⚠️ | Runs but returns 0.0 for everything (no trained models) |
| D3 Network Topology | ✅ | Renders 50 nodes with colors, drag, click — all mock data |
| Recharts Trust Timeline | ✅ | Animates with mock data + responds to WebSocket alerts |
| CUSUM Drift Heatmap | ✅ | Renders with synthetic Tue/Wed night attack pattern |
