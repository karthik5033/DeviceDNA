# DeviceDNA: Comprehensive Codebase & Architecture Guide for AI

This document is a deep-dive technical reference for the **DeviceDNA** platform. It is designed to provide an AI agent with absolute context over the architecture, data structures, machine learning methodologies, and implementation details of the entire project.

---

## 1. Executive Summary
**DeviceDNA** is a real-time, zero-trust IoT/OT network security platform. It ingests simulated network telemetry, processes it through a sophisticated 5-Pillar Machine Learning "Trust Engine," and dynamically calculates a continuous **Trust Score (0-100)** for every device in the network. The platform features an **Autonomous Response Engine** that enforces network quarantine based on trust degradation, and an **Explainability Engine** that generates Threat Intelligence Briefs (TIBs).

---

## 2. Core Architecture & Tech Stack

### A. Backend (`/backend`)
- **Web Framework:** FastAPI (Python 3.9+) with Uvicorn (ASGI).
- **Message Broker:** Apache Kafka (via `aiokafka`). Topic: `raw-flows`.
- **Time-Series DB:** InfluxDB v2 (via `influxdb-client-python`). Stores real-time trust scores and metrics.
- **Relational DB:** PostgreSQL. Managed via SQLAlchemy (AsyncSession) and `asyncpg`. Stores `Alert` objects and HITL (Human-in-the-Loop) interactions.
- **In-Memory Cache:** Redis (via `redis.asyncio`). Stores recent device states, moving averages for peer comparisons, and trust decay history.
- **Real-Time Pub/Sub:** `python-socketio` (ASGI mode) mounted into FastAPI to stream `trust_update` and `new_alert` events to the frontend.

### B. Frontend (`/frontend`)
- **Framework:** Next.js (App Router), React, TypeScript.
- **Styling:** Tailwind CSS, Framer Motion (micro-animations), Lucide React (icons).
- **State/Sockets:** `socket.io-client` connects to the FastAPI backend on port `8000`.
- **Visualizations:** Recharts (Trust Score trajectories), Force-Graph (Network Topology).

### C. Network Simulator (`/simulator`)
- A custom Python script that generates mock IoT telemetry.
- Represents a fleet of devices with profiles (`camera`, `sensor`, `thermostat`, `industrial`, `medical`, `access_control`).
- Injects periodic threat scenarios (Botnet C2, Data Exfiltration, Network Scanning).

---

## 3. Data Flow & Schemas

### 1. Raw Telemetry Ingestion (Kafka)
The simulator pushes JSON payloads to the `raw-flows` Kafka topic at a high frequency (e.g., 100 flows/sec).
```json
{
  "flow_id": "uuid",
  "src_ip": "192.168.1.10",
  "dst_ip": "8.8.8.8",
  "src_port": 45123,
  "dst_port": 443,
  "protocol": "TCP",
  "bytes": 1500,
  "packets": 5,
  "duration": 0.5,
  "timestamp": "2026-06-03T20:10:00Z",
  "device_id": "SIM-001",
  "device_class": "camera"
}
```

### 2. Feature Extraction
In `backend/app/services/telemetry.py`, raw flows are converted into a flat **14-Dimensional Feature Vector**:
`[total_flows, total_bytes, total_packets, avg_packet_size, ... unique_dst_ips, unique_dst_ports, ext_int_ratio, active_hours_bitmap, burst_freq, etc.]`

### 3. Trust Score Redis Schema
Redis stores the most recent state for fast peer comparisons and decay calculations. Key: `trust:{device_id}`.
```json
{
  "score": 85.5,
  "raw_score": 90.0,
  "decay_multiplier": 0.95,
  "device_id": "SIM-001",
  "device_class": "camera",
  "timestamp": "2026-06-03T20:10:00Z",
  "vae_score": 0.05,
  "if_score": 0.12,
  "lstm_score": 0.08,
  "gnn_score": 0.02,
  "ensemble_score": 0.07,
  "policy_penalty": 0.0,
  "peer_penalty": 0.0,
  "penalty": 0.1
}
```

---

## 4. The 5-Pillar Machine Learning Trust Engine

Implemented in `backend/app/services/trust_engine.py` (`TrustScoreEngine`), this engine orchestrates 5 independent algorithms to calculate a penalty, which is subtracted from a base score of 100.

### Pillar 1: Digital Twin (GMVAE) - Weight: 35%
- **Model:** Gaussian Mixture Variational Autoencoder (PyTorch).
- **Purpose:** Learns the complex, multi-modal probability distribution of "normal" behavior for specific device classes. 
- **Mechanism:** Passes the 14D feature vector through encoder/decoder. High reconstruction loss translates to a high penalty.

### Pillar 2: Anomaly Ensemble - Weight: 25%
A sub-ensemble combining structural, temporal, and statistical models.
- **Isolation Forest (IF - 60% of ensemble):** Scikit-learn model. Good at finding statistical outliers in the 14D space.
- **LSTM (20% of ensemble):** PyTorch sequence model. Looks at a rolling window (e.g., last 12 flow snapshots) to predict the next state. Deviations from the predicted sequence indicate temporal anomalies (e.g., unexpected beaconing).
- **Graph Neural Network (GraphSAGE - 20% of ensemble):** PyTorch Geometric model. Analyzes the `src_ip -> dst_ip` communication edges. Identifies structural anomalies like lateral movement.

### Pillar 3: Drift Intelligence (CUSUM) - Weight: 20%
- **Model:** Cumulative Sum Control Chart (Statistical).
- **Purpose:** Detects "low and slow" attacks, like stealthy data exfiltration, which evade sudden-spike detection models (like IF). It tracks subtle, sustained shifts in mean byte transfer rates over time.

### Pillar 4: Policy Conformance - Weight: 15%
- **Mechanism:** Hardcoded heuristic rules defined per `device_class` (e.g., Cameras must have `ext_int_ratio < 0.6` and `bytes_sent < 100000`). Failure to conform linearly increases the penalty.

### Pillar 5: Peer Comparison - Weight: 5%
- **Mechanism:** Retrieves scores of all other devices of the same `device_class` from Redis. If a device's preliminary score deviates significantly from the class mean, an additional penalty is applied.

---

## 5. Advanced Mechanics

### A. Trust Decay & Recovery (`backend/app/services/trust_decay.py`)
- **Decay:** If anomalies trigger frequently in a 60-minute rolling window, a `decay_multiplier` (e.g., `0.40`) is applied to drastically drop the final trust score. 
- **Recovery:** Devices do not regain trust instantly. The `recovery_manager.py` enforces a slow, asynchronous recovery process.

### B. Explainability Engine (TIB) (`backend/app/ml/explainability/tib_generator.py`)
- When a score drops severely (Critical/High alert), the backend generates a **Threat Intelligence Brief (TIB)**.
- Uses **SHAP (SHapley Additive exPlanations)** via a surrogate model to determine exactly *which* features (e.g., `external_traffic_ratio`) contributed most to the anomaly. 
- This translates mathematical model outputs into human-readable SOC explanations.

### C. Autonomous Response Engine (`backend/app/services/response_engine.py`)
Automatically maps trust scores to actionable network policies:
- **Tier 1 (80-100):** Trusted. Standard operation.
- **Tier 2 (60-80):** Guarded. Triggers automated Deep Packet Inspection (DPI) logging.
- **Tier 3 (40-60):** Suspicious. Enforces aggressive rate limiting.
- **Tier 4 (<40):** Critical. Enqueues a network quarantine/isolation command. Requires SOC Analyst approval via the dashboard.

---

## 6. Implementation Bottlenecks & Fixes Applied

To assist with further development, the AI should be aware of a critical bottleneck that was recently resolved:
- **Event Loop Starvation:** Because Kafka delivers 100 flows/sec, executing the heavy PyTorch inference (GMVAE, LSTM, GNN) for *every single flow* sequentially blocked the `asyncio` event loop. This caused `uvicorn` to hang and WebSockets to fail.
- **The Solution:** In `backend/app/services/telemetry.py`, the ML inference (`TrustScoreEngine.evaluate_device`) is throttled (currently 1 in 10 flows) using a modulus counter (`self.flow_count % 10 != 0`). Additionally, blocking file I/O operations inside the async loop were removed. The UI still receives `telemetry_ping` updates for all flows, but the CPU bottleneck is bypassed.

---

## 7. Frontend Integration details

### Dashboard (`frontend/app/dashboard/page.tsx`)
1. **Initial Load:** Fetches historical device trust data from `/api/trust/devices`.
2. **WebSocket Lifecycle:** Connects to `http://localhost:8000`. Listens for:
   - `telemetry_ping`: Animates particles on the Force-Graph.
   - `trust_update`: Updates the Recharts line charts and the top metric cards.
   - `new_alert`: Pushes a notification into the Alert Feed.
3. **Action Triggers:** SOC analysts can click "Isolate Device" which emits an `isolate_device` WS event back to the server, mimicking SDN (Software Defined Networking) integration.

---
*End of Comprehensive Context Document.*
