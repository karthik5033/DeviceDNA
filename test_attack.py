import requests
import random
import time

for _ in range(50):
    try:
        requests.get('http://127.0.0.1:8000/api/health', timeout=1)
    except:
        pass
    port = random.randint(8000, 9000)
    try:
        requests.get(f'http://127.0.0.1:{port}/', timeout=1)
    except:
        pass
