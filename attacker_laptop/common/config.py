import os
import sys

# Add backend directory to path so we can import simulator components if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
TOPIC_NAME = "raw-flows"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "http://localhost:8000")

# Import the FLEET definition dynamically from the simulator
try:
    from simulator.device_profiles import FLEET
except ImportError:
    FLEET = []
