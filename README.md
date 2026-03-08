# DeviceDNA: AI-Powered IoT Cybersecurity Intelligence Platform

DeviceDNA is an enterprise-grade IoT cybersecurity platform that utilizes deep learning (Digital Twins, VAEs, Graph Neural Networks, and LSTMs) to mathematically model and secure IoT network topologies in real-time.

## System Architecture
The platform is composed of 4 main layers:
1. **Data Infrastructure** (Docker: PostgreSQL, InfluxDB, Redis, Kafka, Zookeeper)
2. **Telemetry Simulator** (Python app generating mock realistic IoT fleet data & attacks)
3. **ML Backend API** (FastAPI running PyTorch/Scikit-learn detection models)
4. **SOC Dashboard** (Next.js 14 App Router frontend with D3.js / Recharts visualizations)

---

## 🚀 How to Run the Project

### Prerequisites
- **Git**
- **Docker Desktop** (must be running WSL2 under the hood for Windows)
- **Node.js** (v18+ recommended)
- **Python** (v3.11+ recommended)

### Step 1: Start the Data Infrastructure
Ensure Docker Desktop is open and running.
Open a terminal in the root `DeviceDNA` directory:
```bash
docker compose up -d
```
This spins up Kafka (message broker), PostgreSQL (relational DB), InfluxDB (time-series DB), and Redis (caching layer). Wait about 30 seconds for Kafka to fully initialize.

### Step 2: Start the FastAPI Backend
Open a new terminal in the `backend` directory:
```bash
cd backend
python -m venv venv
# Activate the virtual environment:
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```
The Backend API will be available at `http://localhost:8000`.

### Step 3: Start the Telemetry Simulator
The simulator generates network traffic and injects attacks (C2 Botnets, Lateral Movement, etc.) directly into Kafka.
Open a new terminal in the `backend` directory:
```bash
cd backend
# Activate the virtual environment
.\venv\Scripts\activate
# Run the simulator
python -m simulator.main
```
You should see logs firing indicating simulated network flows are being streamed.

### Step 4: Start the Next.js SOC Dashboard
Open a new terminal in the `frontend` directory:
```bash
cd frontend
npm install
npm run dev
```
The SOC Dashboard will be available at `http://localhost:3000`.

---

## 🧠 Accessing the Platform
Once everything is running, open your web browser to:
[http://localhost:3000](http://localhost:3000)

Click **Enter SOC Dashboard** to view the live ML evaluations, the force-directed D3 network map, and the real-time CUSUM drift matrices!

## Project Structure
* `/backend/app/ml/` - Contains the logic for the VAE Digital Twin, Isolation Forests, LSTM, and GNN models.
* `/backend/simulator/` - The engine responsible for spawning 50 virtual IoT devices and generating their baseline telemetry and subsequent threat anomalies.
* `/frontend/app/` - The full Next.js App Router providing the SOC Visual Interface.
* `/docker-compose.yml` - Infrastructure orchestration. 
