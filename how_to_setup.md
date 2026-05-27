# 🧬 DeviceDNA — Setup Guide

> **Version**: 1.0 &nbsp;|&nbsp; **Last Updated**: May 2026  
> A complete guide for cloning, configuring, and running the DeviceDNA IoT Cybersecurity Platform on your local machine.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Environment Variables](#3-environment-variables)
4. [Option A — One-Command Docker Setup (Recommended)](#4-option-a--one-command-docker-setup-recommended)
5. [Option B — Manual / Local Development Setup](#5-option-b--manual--local-development-setup)
6. [Accessing the Platform](#6-accessing-the-platform)
7. [Project Architecture Overview](#7-project-architecture-overview)
8. [Training the ML Models (Optional)](#8-training-the-ml-models-optional)
9. [Troubleshooting](#9-troubleshooting)
10. [Useful Commands Cheat Sheet](#10-useful-commands-cheat-sheet)

---

## 1. Prerequisites

Make sure you have the following installed before proceeding:

| Tool | Minimum Version | Download Link |
|------|----------------|---------------|
| **Git** | 2.30+ | [git-scm.com](https://git-scm.com/) |
| **Docker Desktop** | 4.x+ (Engine 24+) | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Docker Compose** | v2 (bundled with Desktop) | Included with Docker Desktop |
| **Node.js** *(manual setup only)* | 18 or 20 LTS | [nodejs.org](https://nodejs.org/) |
| **Python** *(manual setup only)* | 3.11+ | [python.org](https://www.python.org/) |

> [!TIP]
> **If you only want to demo/run the project**, you just need **Git + Docker Desktop**. The Docker Compose file handles everything (Python, Node, databases) inside containers. Node.js and Python are only needed if you want to develop locally without Docker.

### System Requirements

- **RAM**: 8 GB minimum (16 GB recommended — PyTorch + Kafka + InfluxDB are memory-heavy)
- **Disk**: ~5 GB free (Docker images + trained ML model weights)
- **OS**: Windows 10/11, macOS 12+, or Linux (Ubuntu 22.04+ recommended)

---

## 2. Clone the Repository

```bash
git clone https://github.com/karthik5033/DeviceDNA.git
cd DeviceDNA
```

---

## 3. Environment Variables

The project ships with a `.env` file in the root directory. **This file is git-ignored**, so you must create it manually after cloning.

Create a file named `.env` in the project root (`DeviceDNA/.env`) with the following contents:

```env
# ═══════════════════════════════════════════
#         DeviceDNA Environment Config
# ═══════════════════════════════════════════

# ── Database (PostgreSQL) ──
DATABASE_URL=postgresql://devicedna:devicedna_password@localhost:5432/devicedna

# ── Time-Series Database (InfluxDB) ──
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=super-secret-influx-token-123
INFLUXDB_ORG=devicedna_org
INFLUXDB_BUCKET=telemetry

# ── Cache & PubSub (Redis) ──
REDIS_URL=redis://localhost:6379/0

# ── Stream Broker (Kafka) ──
KAFKA_BROKER_URL=localhost:29092

# ── Frontend ──
NEXT_PUBLIC_WS_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000

# ── API Security ──
SECRET_KEY=devicedna_secret_dev
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

> [!IMPORTANT]
> The `INFLUXDB_TOKEN` value **must match** the `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN` in `docker-compose.yml`. If you change one, change both. The default value (`super-secret-influx-token-123`) works out of the box.

---

## 4. Option A — One-Command Docker Setup (Recommended)

This is the easiest way to get the entire platform running. Docker Compose will spin up **9 containers**: PostgreSQL, InfluxDB, Redis, Zookeeper, Kafka, a database seeder, the FastAPI backend, the traffic simulator, and the Next.js frontend.

### Step 1: Start everything

```bash
docker-compose up --build
```

That's it. Docker will:
1. Pull the required images (first run takes 5–10 min depending on internet speed)
2. Start PostgreSQL, InfluxDB, Redis, Kafka, and Zookeeper
3. Run the **seeder** to populate the database with 50 simulated IoT devices
4. Start the **FastAPI backend** (port `8000`) with Socket.IO WebSockets
5. Start the **traffic simulator** that feeds live synthetic network flows through Kafka
6. Start the **Next.js frontend** (port `3000`)

### Step 2: Wait for healthy status

In a new terminal, check the health of all services:

```bash
docker-compose ps
```

You should see all services as `Up` or `Up (healthy)`. The `seeder` will show `Exit 0` — that's expected (it runs once and exits).

### Step 3: Open the dashboard

Navigate to **[http://localhost:3000](http://localhost:3000)** in your browser.

### Stopping the platform

```bash
docker-compose down
```

To also **wipe all data** (databases, volumes):

```bash
docker-compose down -v
```

---

## 5. Option B — Manual / Local Development Setup

Use this if you want hot-reload, debugger support, or want to modify code without rebuilding containers.

### 5.1 Start Infrastructure Services Only

Spin up only the databases and message broker (not the app containers):

```bash
docker-compose up postgres influxdb redis zookeeper kafka -d
```

Wait ~30 seconds for Kafka and PostgreSQL health checks to pass.

### 5.2 Backend Setup

```bash
cd backend

# Create a Python virtual environment
python -m venv venv

# Activate it
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> **PyTorch** is included in `requirements.txt` (`torch==2.2.2`). On a CPU-only machine this will install the CPU variant automatically (~800 MB download). If you have an NVIDIA GPU and want CUDA support, install PyTorch manually first following [pytorch.org/get-started](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

### 5.3 Seed the Database

Populate PostgreSQL with 50 simulated devices and initial alert data:

```bash
# Make sure you're in the backend/ directory with venv activated
python -m scripts.seed_demo_data
```

You should see: `Demo data seeded successfully`

### 5.4 Start the Backend Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables hot-reload for development. The backend will:
- Initialize PostgreSQL tables (via SQLAlchemy)
- Connect to Redis, InfluxDB, and Kafka
- Start the Kafka consumer for real-time telemetry processing
- Expose the REST API and Socket.IO WebSocket on port **8000**

### 5.5 Start the Traffic Simulator

Open a **second terminal** (with the venv activated) in the `backend/` directory:

```bash
python -m simulator.main
```

This generates synthetic network flows for 50 devices across 6 device classes (camera, sensor, thermostat, access_control, medical, industrial) and publishes them to Kafka. The trust engine will process them in real-time.

### 5.6 Frontend Setup

Open a **third terminal**:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the dev server
npm run dev
```

The frontend starts on **[http://localhost:3000](http://localhost:3000)** with hot-reload enabled.

> [!NOTE]
> The Next.js config contains API rewrites that proxy `/api/*` requests to `http://127.0.0.1:8000/api/*`. This means the frontend and backend communicate seamlessly on different ports without CORS issues during local development.

---

## 6. Accessing the Platform

Once everything is running, open your browser:

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | [http://localhost:3000](http://localhost:3000) | Main landing page with animated KPIs |
| **SOC Overview** | [http://localhost:3000/dashboard](http://localhost:3000/dashboard) | Network topology, trust timeline, live event log |
| **Device Deep-Dive** | [http://localhost:3000/dashboard/node/SIM-0001](http://localhost:3000/dashboard/node/SIM-0001) | Radial trust gauge, CUSUM heatmap, SHAP briefs |
| **Alerts** | [http://localhost:3000/dashboard/alerts](http://localhost:3000/dashboard/alerts) | Real-time threat alert feed |
| **Policies** | [http://localhost:3000/dashboard/policies](http://localhost:3000/dashboard/policies) | NLP-based policy engine |
| **Predictive Risk** | [http://localhost:3000/dashboard/predict](http://localhost:3000/dashboard/predict) | LSTM forecasting & CUSUM drift heatmap |
| **Replay / Forensics** | [http://localhost:3000/dashboard/replay](http://localhost:3000/dashboard/replay) | Historical incident timeline scrubber |
| **Topology** | [http://localhost:3000/dashboard/topology](http://localhost:3000/dashboard/topology) | Full-page D3 network topology map |
| **Backend API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | FastAPI auto-generated Swagger UI |
| **Backend Health** | [http://localhost:8000/api/health](http://localhost:8000/api/health) | Quick health check endpoint |
| **InfluxDB UI** | [http://localhost:8086](http://localhost:8086) | InfluxDB admin panel (user: `devicedna` / pass: `devicedna_password`) |

---

## 7. Project Architecture Overview

```
DeviceDNA/
├── backend/
│   ├── app/
│   │   ├── api/routes/         # FastAPI route handlers (trust, alerts, policy, response)
│   │   ├── db/                 # Database clients (PostgreSQL, InfluxDB, Redis)
│   │   ├── ml/                 # ML modules (explainability/SHAP, NLP policy parser)
│   │   ├── services/           # Core business logic (trust_engine, response_engine, telemetry)
│   │   └── main.py             # App entrypoint (FastAPI + Socket.IO ASGI wrapper)
│   ├── db_init/init.sql        # PostgreSQL schema initialization
│   ├── models_trained/         # Pre-trained model weights (.pt, .joblib, .json)
│   ├── simulator/              # Synthetic IoT traffic generator
│   ├── training/               # Model training scripts (VAE, IF, LSTM, GNN)
│   ├── scripts/                # Utility scripts (seeder, InfluxDB tests)
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── app/                    # Next.js 14 App Router pages
│   ├── components/             # React components (layout, visualizations)
│   ├── lib/                    # Utility functions
│   ├── store/                  # Zustand state management
│   └── package.json            # Node dependencies
├── docs/                       # Documentation, reports, PRDs
├── docker-compose.yml          # Full-stack orchestration
├── .env                        # Environment variables (create manually)
└── README.md                   # Project overview
```

### Data Flow Pipeline

```
IoT Simulator → Kafka → Telemetry Service → ML Trust Engine (5 Pillars) → Redis Cache
                                                    ↓
                                              PostgreSQL (Alerts)
                                              InfluxDB (Time-Series)
                                                    ↓
                                          Socket.IO WebSocket → React Frontend
```

### The 5 ML Pillars

| # | Pillar | Model | Purpose |
|---|--------|-------|---------|
| 1 | Digital Twin | VAE (Variational Autoencoder) | Detects deviation from learned normal baseline |
| 2 | Structural Anomaly | Isolation Forest | Identifies outlier feature vectors per device class |
| 3 | Temporal Forecasting | LSTM | Predicts next feature vector; flags temporal anomalies |
| 4 | Spatial / Lateral | GNN (GraphSAGE) | Detects anomalous communication patterns in device graphs |
| 5 | Statistical Drift | CUSUM | Catches slow, stealthy data exfiltration over time |

---

## 8. Training the ML Models (Optional)

Pre-trained model weights are already included in `backend/models_trained/`. You only need to retrain if you want to experiment with the models or use a different dataset.

```bash
cd backend

# Activate your virtual environment first
# Then run each training script:

python -m training.train_vae           # Train VAE autoencoders (one per device)
python -m training.train_isolation_forest  # Train Isolation Forest (one per device class)
python -m training.train_lstm          # Train shared LSTM forecaster
python -m training.train_gnn           # Train shared GNN (GraphSAGE)
```

> [!WARNING]
> Training runs against synthetic simulator data. The models in the repo are trained on simulated traffic — they are **not production-grade**. For real deployment, ingest a proper dataset (e.g., IoT-23, UNSW-NB15) and retrain.

---

## 9. Troubleshooting

### ❌ `docker-compose up` fails with port conflicts

Another service is already using one of the required ports. Check with:

```bash
# Windows
netstat -ano | findstr "5432 8086 6379 9092 8000 3000"

# macOS / Linux
lsof -i :5432 -i :8086 -i :6379 -i :9092 -i :8000 -i :3000
```

Stop the conflicting service or change the port mapping in `docker-compose.yml`.

### ❌ Backend crashes with `ModuleNotFoundError`

Make sure your virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

### ❌ Frontend shows "Failed to fetch" or blank dashboard

1. Make sure the **backend is running** on port 8000
2. Check that your `.env` has the correct `NEXT_PUBLIC_WS_URL=http://localhost:8000`
3. Open the browser DevTools console (F12) and check for WebSocket connection errors

### ❌ No devices or alerts appearing

The simulator might not be running. Check:

```bash
# Docker
docker-compose logs simulator

# Manual
# Make sure `python -m simulator.main` is running in a separate terminal
```

### ❌ InfluxDB connection refused

InfluxDB takes a few seconds to initialize on first run. Wait 10–15 seconds and try again. Verify it's running:

```bash
docker-compose logs influxdb
```

### ❌ Kafka broker not available

Kafka depends on Zookeeper. Both need ~15 seconds to fully start. If you see `NoBrokersAvailable`, wait and retry. You can verify:

```bash
docker-compose logs kafka | tail -20
```

### ❌ `CRLF` warnings during `git add`

These are harmless line-ending warnings (Windows vs Unix). You can suppress them with:

```bash
git config core.autocrlf true
```

---

## 10. Useful Commands Cheat Sheet

| Action | Command |
|--------|---------|
| **Start everything (Docker)** | `docker-compose up --build` |
| **Start in background** | `docker-compose up -d --build` |
| **View logs (all)** | `docker-compose logs -f` |
| **View backend logs** | `docker-compose logs -f backend` |
| **View simulator logs** | `docker-compose logs -f simulator` |
| **Stop everything** | `docker-compose down` |
| **Stop + wipe data** | `docker-compose down -v` |
| **Rebuild single service** | `docker-compose up --build backend` |
| **Check service health** | `docker-compose ps` |
| **Enter backend container** | `docker-compose exec backend bash` |
| **Reset PostgreSQL** | `docker-compose down -v && docker-compose up -d postgres` |
| **Backend health check** | `curl http://localhost:8000/api/health` |
| **Run seeder manually** | `cd backend && python -m scripts.seed_demo_data` |
| **Run simulator manually** | `cd backend && python -m simulator.main` |
| **Frontend dev server** | `cd frontend && npm run dev` |
| **Backend dev server** | `cd backend && uvicorn app.main:app --reload --port 8000` |

---

## 🎉 You're Ready!

Once all services are green, head over to **[http://localhost:3000](http://localhost:3000)** and you'll see the DeviceDNA landing page. Click **"Enter SOC Dashboard"** to access the main operations center with live trust scores, network topology, alerts, and autonomous response panels.

If you run into any issues not covered here, check the [docs/](docs/) folder for additional documentation, or open an issue on the GitHub repository.

---

*Built with ❤️ by the DeviceDNA team*
