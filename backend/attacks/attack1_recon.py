import sys
import time
import json
import redis
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Attack 1 - Stealth Reconnaissance (nmap + injection)")
    parser.add_argument("subnet", nargs="?", default="192.168.43.0/24", help="Target subnet for nmap scan (e.g. 192.168.43.0/24)")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    # Target the new physical hardware sensors for the attack
    cameras = ['dht11_sensor', 'mq135_sensor', 'ir_sensor']
    attack_payload = json.dumps({
        "type": "recon",
        "intensity": 0.3
    })

    print('[ATTACK 1] Stealth reconnaissance starting...')
    print(f'[ATTACK 1] Injecting recon behavior into cameras: {", ".join(cameras)}')
    
    for cam_id in cameras:
        r.set(f"attack_state:{cam_id}", attack_payload)

    print(f'[ATTACK 1] Launching nmap scan against {args.subnet}...')
    print(f'         Command: nmap -sS -T2 -p 1-1024 {args.subnet}')
    
    nmap_process = None
    try:
        nmap_process = subprocess.Popen(
            ["nmap", "-sS", "-T2", "-p", "1-1024", args.subnet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except FileNotFoundError:
        print("[!] Warning: 'nmap' is not installed or not in PATH. Skipping the physical scan.")
        print("[!] The backend simulator injection will still run for 120 seconds.")

    print('[ATTACK 1] Waiting 120 seconds for anomaly detection models to process the telemetry...')
    try:
        time.sleep(120)
    except KeyboardInterrupt:
        print('\n[ATTACK 1] Interrupted by user.')

    print('[ATTACK 1] Cleaning up injected behavior...')
    for cam_id in cameras:
        r.delete(f"attack_state:{cam_id}")

    if nmap_process:
        print('[ATTACK 1] Terminating nmap if still running...')
        nmap_process.terminate()
        try:
            out, err = nmap_process.communicate(timeout=5)
            print(f"[ATTACK 1] Nmap Output:\n{out}")
        except subprocess.TimeoutExpired:
            nmap_process.kill()

    print('[ATTACK 1] Recon complete')

if __name__ == "__main__":
    main()
