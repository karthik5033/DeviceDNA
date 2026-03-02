import socketio
import logging
import asyncio

logger = logging.getLogger(__name__)

# Create the async Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

@sio.event
async def connect(sid, environ, auth):
    logger.info(f"Frontend Client connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Frontend Client disconnected: {sid}")

@sio.event
async def isolate_device(sid, data):
    device = data.get('device')
    logger.warning(f"🚨 SOC MANUALLY ISOLATING DEVICE {device} 🚨")
    # Simulate a network propagation delay to SDN
    await asyncio.sleep(1.5)
    # Success broadcast
    await sio.emit('device_isolated', {'device': device, 'status': 'success'})

