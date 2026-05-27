# How to Run the DeviceDNA Platform

DeviceDNA is a full-stack platform consisting of three main parts: the core infrastructure, the Python backend, and the Next.js frontend. 

To run the entire system from scratch, follow these three steps in separate terminal windows:

### 1. Start the Core Infrastructure (Docker)

The application relies on several databases and stream processing services (Kafka, Postgres, Redis, InfluxDB). Start these first from the root directory:

```powershell
cd d:\coding_files\Projects\DeviceDNA
docker-compose up -d
```

### 2. Start the Backend API (FastAPI)

The backend handles machine learning inference, database routing, and WebSocket telemetry streams.

Open a new terminal, activate the python environment, and start the API:

```powershell
cd d:\coding_files\Projects\DeviceDNA\backend
.\venv\Scripts\activate
uvicorn app.main:app --reload
```
*(This will start the backend on `http://localhost:8000`)*

### 3. Start the Frontend Dashboard (Next.js)

The cinematic UI is powered by Next.js. Open a final terminal window and start the frontend:

```powershell
cd d:\coding_files\Projects\DeviceDNA\frontend
npm run dev
```

### 4. View the Dashboard

Once everything is running, open your web browser and navigate directly to:

**👉 [http://localhost:3000/dashboard](http://localhost:3000/dashboard)**

*(Note: The `http://localhost:3000` homepage also has a button to enter the dashboard).*
