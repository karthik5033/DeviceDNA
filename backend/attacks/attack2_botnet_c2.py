"""
Attack 2 — Botnet Command & Control (C2) Beaconing
===================================================

WHAT THIS ATTACK SIMULATES:
  A compromised IoT camera has been enrolled into a botnet. The attacker has
  planted a backdoor that periodically "phones home" to 3 rotating Command &
  Control (C2) servers on port 4444 every 30 seconds, sending a tiny 128-byte
  heartbeat ("I'm alive") packet.

HOW IT WORKS:
  1. The script injects a Redis key `attack_state:<device_id>` with type="beacon"
     for each target device.
  2. The simulator's traffic_generator reads ATTACK_STATE on every cycle and
     generates anomalous C2-like flows instead of normal traffic:
       - Protocol: TCP on port 4444 (highly suspicious for an IoT camera)
       - Destination: 3 rotating fake C2 IPs (203.0.113.x, 198.51.100.x)
       - Payload: Tiny 128-byte packets (classic beacon signature)
       - Pattern: Periodic, low-jitter interval (beaconing fingerprint)
  3. The Trust Engine's ML pipeline detects this via:
       - VAE Digital Twin: Reconstruction error spikes because port 4444
         and external C2 IPs are completely outside the camera's learned profile.
       - Isolation Forest: The tiny packet size + unknown port combination
         is a statistical outlier.
       - LSTM Temporal: The periodic, clock-like timing of the beacons is
         detected as a non-human sequence pattern.
  4. The trust score drops rapidly, the node turns RED on the dashboard,
     and a CRITICAL alert is generated.

TARGETS:
  - 2 Virtual cameras (SIM-0009, SIM-0012) — simulated C2 enrollment
  - 1 Physical sensor (dht11_sensor) — shows cross-device infection spread
  - 1 Virtual sensor (SIM-0003) — additional infected node

DURATION: 5 minutes (300 seconds), then auto-cleanup.
"""

import sys
import time
import json
import redis
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Attack 2 - Botnet C2 Beaconing")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--duration", type=int, default=300, help="Attack duration in seconds")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    # Verify Redis connection before starting
    try:
        r.ping()
        print(f'[ATTACK 2] Connected to Redis at {args.redis_host}:{args.redis_port}')
    except redis.exceptions.ConnectionError:
        print(f'[ERROR] Cannot connect to Redis at {args.redis_host}:{args.redis_port}')
        print(f'        Make sure Docker is running: docker-compose up -d redis')
        print(f'        Or if Redis is local: redis-server')
        sys.exit(1)

    # Target devices: cameras + a physical sensor to show infection spread
    targets = [
        'SIM-0009',       # Virtual Camera
        'SIM-0012',       # Virtual Camera
        'dht11_sensor',   # Physical Sensor (cross-device infection)
        'SIM-0003',       # Virtual Sensor
    ]

    print('=' * 60)
    print('  ATTACK 2 -- Two-Stage Botnet & DDoS Attack')
    print('=' * 60)
    print()
    print(f'[ATTACK 2] Targets: {", ".join(targets)}')
    print(f'[ATTACK 2] Total Duration: {args.duration} seconds')
    print()
    
    # -------------------------------------------------------------------------
    # STAGE 1: BOTNET C2 ENROLLMENT (BEACONING)
    # -------------------------------------------------------------------------
    print('>>> STAGE 1: Botnet C2 Enrollment (Stealth Beaconing)')
    print('[ATTACK 2] Injecting C2 beacon behavior into compromised devices...')

    beacon_payload = json.dumps({
        "type": "beacon",
        "intensity": 0.7,
        "c2_servers": ["203.0.113.4", "198.51.100.22", "192.0.2.77"],
        "c2_port": 4444,
        "beacon_interval_sec": 30,
        "beacon_payload_bytes": 128
    })

    for device_id in targets:
        r.set(f"attack_state:{device_id}", beacon_payload)
        print(f'  [+] {device_id} -- C2 backdoor implanted (beacon to 203.0.113.4:4444)')

    print()
    print(f'[ATTACK 2] Beaconing active. Devices are now phoning home every 30 seconds.')
    print(f'[ATTACK 2] Waiting 120 seconds for Phase 1 to develop...')
    print()
    print('  Expected Phase 1 Detection:')
    print('    * VAE Digital Twin  -> HIGH anomaly (unknown port 4444 + external C2 IPs)')
    print('    * LSTM Temporal     -> HIGH anomaly (periodic 30s beaconing pattern)')

    try:
        time.sleep(120)
    except KeyboardInterrupt:
        print('\n[ATTACK 2] Interrupted by user.')
        cleanup(r, targets)
        return

    # -------------------------------------------------------------------------
    # STAGE 2: MASSIVE DDOS FLOOD
    # -------------------------------------------------------------------------
    print()
    print('>>> STAGE 2: Command Received -> VOLUMETRIC DDoS FLOOD')
    print('[ATTACK 2] Botmaster has issued the attack command.')
    print('[ATTACK 2] Switching infected devices into UDP flood mode against 198.51.100.99...')

    ddos_payload = json.dumps({
        "type": "ddos",
        "intensity": 1.0,
        "ddos_target_ip": "198.51.100.99"
    })

    for device_id in targets:
        r.set(f"attack_state:{device_id}", ddos_payload)
        print(f'  [!!!] {device_id} -- Unleashing massive UDP flood')

    print()
    print(f'[ATTACK 2] DDoS active! Waiting remaining 180 seconds...')
    print()
    print('  Expected Phase 2 Detection:')
    print('    * Isolation Forest  -> CRITICAL anomaly (massive byte/packet spike)')
    print('    * CUSUM Drift       -> CRITICAL anomaly (severe deviation from baseline volume)')

    try:
        time.sleep(180)
    except KeyboardInterrupt:
        print('\n[ATTACK 2] Interrupted by user.')

    cleanup(r, targets)
    
    print()
    print('[ATTACK 2] Two-Stage Botnet attack complete.')
    print('[ATTACK 2] Devices will gradually recover their trust scores.')

def cleanup(r, targets):
    print('\n[ATTACK 2] Cleaning up -- removing implants from Redis...')
    for device_id in targets:
        r.delete(f"attack_state:{device_id}")
        print(f'  [-] {device_id} -- Backdoor removed')

if __name__ == "__main__":
    main()
