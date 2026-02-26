from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from contextlib import asynccontextmanager

from app.services.telemetry import TelemetryService
from app.api.routes import trust
from app.db.influxdb import influx_db

telemetry_service = TelemetryService(influx_db)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run the Kafka flow consumer in the background
    await telemetry_service.start()
    yield
    # Shutdown: Clean up connections
    await telemetry_service.stop()
    await influx_db.close()

app = FastAPI(
    title="DeviceDNA API",
    description="Backend API for the DeviceDNA IoT Cybersecurity Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Insert the API routes into the root app
app.include_router(trust.router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "DeviceDNA Backend"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
