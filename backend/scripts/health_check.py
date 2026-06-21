"""
DeviceDNA — Service Health Checker
Run from backend/ folder with venv active:
    python -m scripts.health_check

Checks: PostgreSQL, Redis, InfluxDB, Kafka, FastAPI Backend, MQTT Broker
"""

import sys
import os
import socket
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(name, detail=""):
    tag = f"{GREEN}✓ OK   {RESET}"
    print(f"  {tag} {BOLD}{name:<20}{RESET}  {detail}")

def fail(name, detail=""):
    tag = f"{RED}✗ FAIL {RESET}"
    print(f"  {tag} {BOLD}{name:<20}{RESET}  {detail}")

def warn(name, detail=""):
    tag = f"{YELLOW}⚠ WARN {RESET}"
    print(f"  {tag} {BOLD}{name:<20}{RESET}  {detail}")

def tcp_check(host, port, timeout=2):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False

def check_postgres():
    if not tcp_check("localhost", 5432):
        fail("PostgreSQL", "Port 5432 not reachable — is Docker running?")
        return False
    try:
        import asyncio
        from sqlalchemy import text
        from app.db.postgres import AsyncSessionLocal, engine

        async def _q():
            async with AsyncSessionLocal() as s:
                result = await s.execute(text("SELECT COUNT(*) FROM alerts"))
                count = result.scalar()
                return count
        
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        count = asyncio.run(_q())
        ok("PostgreSQL", f"Port 5432 open | alerts table: {count} rows")
        return True
    except Exception as e:
        warn("PostgreSQL", f"Port open but query failed: {str(e)[:60]}")
        return False

def check_redis():
    if not tcp_check("localhost", 6379):
        fail("Redis", "Port 6379 not reachable — is Docker running?")
        return False
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
        r.ping()
        trust_keys = len(r.keys("trust:*"))
        ok("Redis", f"Port 6379 open | trust keys cached: {trust_keys}")
        return True
    except Exception as e:
        warn("Redis", f"Port open but ping failed: {str(e)[:60]}")
        return False

def check_influxdb():
    if not tcp_check("localhost", 8086):
        fail("InfluxDB", "Port 8086 not reachable — is Docker running?")
        return False
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8086/ping", timeout=2) as resp:
            if resp.status in (200, 204):
                ok("InfluxDB", "Port 8086 open | /ping OK")
                return True
    except Exception as e:
        warn("InfluxDB", f"Port open but ping failed: {str(e)[:60]}")
    return False

def check_kafka():
    if not tcp_check("localhost", 29092):
        fail("Kafka", "Port 29092 not reachable — is Docker running?")
        return False
    ok("Kafka", "Port 29092 open (broker reachable)")
    return True

def check_backend():
    if not tcp_check("localhost", 8000):
        fail("FastAPI Backend", "Port 8000 not reachable — run: uvicorn app.main:app")
        return False
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://localhost:8000/api/health", timeout=3) as resp:
            data = json.loads(resp.read())
            status = data.get("status", "unknown")
            ok("FastAPI Backend", f"Port 8000 open | status: {status}")
            return True
    except Exception as e:
        warn("FastAPI Backend", f"Port open but /api/health failed: {str(e)[:60]}")
        return False

def check_mqtt():
    if not tcp_check("localhost", 1883):
        warn("MQTT Broker", "Port 1883 not reachable — MQTT in simulated mode (OK for dev)")
        return False
    ok("MQTT Broker", "Port 1883 open (Mosquitto running)")
    return True

def check_frontend():
    if not tcp_check("localhost", 3000):
        fail("Frontend", "Port 3000 not reachable — run: npm run dev in frontend/")
        return False
    ok("Frontend", "Port 3000 open | http://localhost:3000")
    return True


def main():
    print()
    print(f"{CYAN}{BOLD}{'='*55}{RESET}")
    print(f"{CYAN}{BOLD}  DeviceDNA — Service Health Check{RESET}")
    print(f"{CYAN}{BOLD}{'='*55}{RESET}")
    print()

    results = {}
    results["PostgreSQL"]   = check_postgres()
    results["Redis"]        = check_redis()
    results["InfluxDB"]     = check_influxdb()
    results["Kafka"]        = check_kafka()
    results["MQTT"]         = check_mqtt()
    results["Backend"]      = check_backend()
    results["Frontend"]     = check_frontend()

    print()
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    if passed == total:
        print(f"  {GREEN}{BOLD}All {total} services healthy! DeviceDNA is fully operational.{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}{passed}/{total} services healthy.{RESET}")
        failed = [k for k, v in results.items() if not v]
        print(f"  {RED}Not running: {', '.join(failed)}{RESET}")
        print()
        print(f"  {CYAN}Quick fix:{RESET}")
        print(f"    docker-compose up -d postgres redis influxdb zookeeper kafka mosquitto")
        print(f"    cd backend && .\\venv\\Scripts\\activate && uvicorn app.main:app --reload")
        print(f"    cd frontend && npm run dev")

    print(f"\n{CYAN}{BOLD}{'='*55}{RESET}")
    print()

if __name__ == "__main__":
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
