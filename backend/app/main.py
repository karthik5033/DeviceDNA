from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import socketio

from contextlib import asynccontextmanager

# Configure root logger so all app.services.* / app.ml.* loggers emit to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

from app.services.telemetry import TelemetryService
from app.services.hardware_registry import registry_maintenance_loop
from app.api.routes import trust, alerts, policy, response, hardware_health, audit, attacks
from app.db.influxdb import influx_db
from app.db.postgres import engine, Base
from app.api.ws import sio
from app.services.sniffer import live_sniffer

import asyncio

telemetry_service = TelemetryService(influx_db)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PostgreSQL tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Startup: Run the Kafka flow consumer in the background
    await telemetry_service.start()
    
    # Startup: Run the Live Packet Sniffer in the background
    live_sniffer.start()
    
    # Startup: Run the Hardware Registry stale check loop
    app.state.registry_task = asyncio.create_task(registry_maintenance_loop())
    
    yield
    # Shutdown: Clean up connections
    app.state.registry_task.cancel()
    live_sniffer.stop()
    await telemetry_service.stop()
    await influx_db.close()

fastapi_app = FastAPI(
    title="DeviceDNA API",
    description="Backend API for the DeviceDNA IoT Cybersecurity Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Insert the API routes into the root app
fastapi_app.include_router(trust.router)
fastapi_app.include_router(alerts.router, prefix="/api")
fastapi_app.include_router(policy.router, prefix="/api")
fastapi_app.include_router(response.router)
fastapi_app.include_router(hardware_health.router, prefix="/api/hardware", tags=["hardware"])
fastapi_app.include_router(audit.router)
fastapi_app.include_router(attacks.router)

# Configure CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/api/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "DeviceDNA Backend"}

# Socket.io ASGI wrapper - Exposed as `app` so default `uvicorn app.main:app` picks up WebSockets
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
