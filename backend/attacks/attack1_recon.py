import sys
import time
import json
import redis
import argparse
import subprocess
import socket
import threading
import ipaddress
import os

def get_local_subnet():
    """Detects the machine's local IP and assumes a /24 subnet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    
    if local_ip == '127.0.0.1':
        return "10.244.239.0/24" # Fallback
        
    parts = local_ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def main():
    parser = argparse.ArgumentParser(description="Attack 1 - Stealth Reconnaissance (nmap + injection)")
    parser.add_argument("--subnet", help="Target subnet for nmap scan. Auto-detected if not provided.")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    target_subnet = args.subnet if args.subnet else get_local_subnet()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    # Target both physical hardware sensors AND specific virtual devices
    targets = [
        # Physical Devices
        'smoke_sensor_1', 'smoke_sensor_2', 'gyro_sensor',
        # Virtual Devices
        'SIM-0005', 'SIM-0015', 'SIM-0030'
    ]
    attack_payload = json.dumps({
        "type": "recon",
        "intensity": 0.3
    })

    print('[ATTACK 1] Stealth reconnaissance starting...')
    print(f'[ATTACK 1] Injecting recon behavior into targets: {", ".join(targets)}')
    
    for cam_id in targets:
        r.set(f"attack_state:{cam_id}", attack_payload)

    print(f'[ATTACK 1] Auto-detected attacker subnet: {target_subnet}')
    
    # Bypass PATH issues by checking default Windows installation locations
    nmap_executable = "nmap"
    if os.name == 'nt':
        for path in [r"C:\Program Files (x86)\Nmap\nmap.exe", r"C:\Program Files\Nmap\nmap.exe"]:
            if os.path.exists(path):
                nmap_executable = path
                break

    print(f'[ATTACK 1] Launching nmap scan against {target_subnet}...')
    print(f'         Command: {nmap_executable} -sS -T2 -p 1-1024 {target_subnet}')
    
    nmap_process = None
    try:
        nmap_process = subprocess.Popen(
            [nmap_executable, "-sS", "-T2", "-p", "1-1024", target_subnet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except FileNotFoundError:
        print("[!] Warning: 'nmap' is not installed or not in PATH.")
        print(f"[ATTACK 1] Falling back to pure-Python TCP scan for {target_subnet}...")
        
        def python_port_scan(subnet):
            network = ipaddress.ip_network(subnet, strict=False)
            ports = [21, 22, 23, 80, 443, 554, 1883, 5000, 8080]
            def scan_ip(ip):
                for port in ports:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(0.2)
                        s.connect((str(ip), port))
                        s.close()
                    except Exception:
                        pass
                    time.sleep(0.05)
            for ip in network.hosts():
                t = threading.Thread(target=scan_ip, args=(ip,), daemon=True)
                t.start()
                time.sleep(0.02)
                
        # Launch fallback scanner in background thread
        scan_thread = threading.Thread(target=python_port_scan, args=(target_subnet,), daemon=True)
        scan_thread.start()

    print('[ATTACK 1] Waiting 300 seconds for anomaly detection models to process the telemetry...')
    try:
        time.sleep(300)
    except KeyboardInterrupt:
        print('\n[ATTACK 1] Interrupted by user.')

    print('[ATTACK 1] Cleaning up injected behavior...')
    for cam_id in targets:
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
