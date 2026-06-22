import argparse
import asyncio
import sys
import logging
import os
import requests
import socket
import threading
import time
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacker_laptop.common.config import API_BASE_URL, FLEET

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AttackOrchestrator")

# Initialize ANSI colors for terminal UI
if os.name == 'nt':
    os.system('') # Enables ANSI rendering on Windows

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# In-memory session trackers
C2_SERVER_ACTIVE = False
EXFIL_SERVER_ACTIVE = False
C2_MUTEX = threading.Lock()
EXFIL_MUTEX = threading.Lock()

def start_c2_listener():
    """Start local socket server on port 4444 to receive C2 beacons."""
    global C2_SERVER_ACTIVE
    with C2_MUTEX:
        if C2_SERVER_ACTIVE:
            return
        C2_SERVER_ACTIVE = True

    def run_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', 4444))
            s.listen(10)
            logger.info(f"{GREEN}[+] C2 Listener server listening on port 4444...{RESET}")
            while C2_SERVER_ACTIVE:
                s.settimeout(2.0)
                try:
                    conn, addr = s.accept()
                    data = conn.recv(1024).decode('utf-8').strip()
                    if data:
                        logger.info(f"\n{RED}🚨 [C2 BEACON RECEIVED] {addr[0]}: {data}{RESET}")
                    conn.close()
                except socket.timeout:
                    continue
        except Exception as e:
            logger.error(f"C2 Listener error: {e}")
        finally:
            s.close()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

def start_exfil_listener():
    """Start local socket server on port 9999 to receive exfiltrated data."""
    global EXFIL_SERVER_ACTIVE
    with EXFIL_MUTEX:
        if EXFIL_SERVER_ACTIVE:
            return
        EXFIL_SERVER_ACTIVE = True

    def run_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', 9999))
            s.listen(10)
            logger.info(f"{GREEN}[+] Exfiltration receiver server listening on port 9999...{RESET}")
            while EXFIL_SERVER_ACTIVE:
                s.settimeout(2.0)
                try:
                    conn, addr = s.accept()
                    data = conn.recv(8192)
                    if data:
                        logger.info(f"\n{RED}🚨 [EXFIL DATA EXTRAPOLATED] {addr[0]}: Received {len(data)} bytes of dump data{RESET}")
                    conn.close()
                except socket.timeout:
                    continue
        except Exception as e:
            logger.error(f"Exfil Listener error: {e}")
        finally:
            s.close()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

def fetch_active_fleet():
    """Fetch live fleet list from backend API or local profiles."""
    print(f"\n{CYAN}[~] Fetching live fleet state from backend...{RESET}")
    try:
        r = requests.get(f"{API_BASE_URL}/api/trust/devices/all", timeout=3)
        if r.status_code == 200:
            devices = r.json()
            table_data = []
            for dev in devices:
                score = dev.get("trust_score") or dev.get("score", 100.0)
                status = dev.get("status", "trusted").upper()
                
                # ANSI colors for status
                status_color = GREEN if status == "TRUSTED" else YELLOW if status == "GUARDED" else RED
                
                table_data.append([
                    dev["id"],
                    dev.get("name", "Unknown"),
                    dev["device_class"],
                    dev["ip_address"],
                    f"{score:.2f}",
                    f"{status_color}{status}{RESET}"
                ])
            
            print(f"\n{BOLD}{CYAN}Live IoT Fleet Profiles ({len(devices)} Devices):{RESET}")
            print(tabulate(
                table_data,
                headers=["Device ID", "Name", "Class", "IP Address", "Trust Score", "Status"],
                tablefmt="simple"
            ))
            return devices
    except Exception as e:
        print(f"{YELLOW}[!] Backend offline ({e}). Using static fleet profile...{RESET}")

    # Fallback to local profile config
    table_data = []
    for dev in FLEET:
        table_data.append([
            dev["id"],
            dev["name"],
            dev["device_class"],
            dev["ip_address"],
            "100.00",
            f"{GREEN}TRUSTED{RESET}"
        ])
    print(tabulate(
        table_data,
        headers=["Device ID", "Name", "Class", "IP Address", "Trust Score", "Status"],
        tablefmt="simple"
    ))
    return FLEET

async def trigger_exploit(target_ip, device_id, attacker_ip, attack_type):
    """Sends a TCP trigger packet to the simulator's Exploit Listener (port 8888)."""
    print(f"{CYAN}[~] Dispatched exploit payload to {target_ip}:8888 for {device_id}...{RESET}")
    try:
        reader, writer = await asyncio.open_connection(target_ip, 8888)
        payload = f"EXPLOIT:{device_id}:{attacker_ip}:{attack_type}\n"
        writer.write(payload.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        print(f"{GREEN}[+] Exploit successfully executed! Compromised node: {device_id}{RESET}")
    except Exception as e:
        print(f"{RED}[x] Exploit connection failed to {target_ip}:8888. Make sure the simulator container is running and healthy. ({e}){RESET}")

def run_nmap_scan(target_ip):
    """Runs a real TCP connect scan using Python's socket interface (cross-platform)."""
    print(f"\n{BOLD}{RED}🚀 Running Attack 1 (Stealth TCP Recon Scan) targeting: {target_ip}{RESET}")
    ports = [21, 22, 23, 80, 443, 1883, 5432, 8080, 8888]
    open_ports = []
    
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        result = s.connect_ex((target_ip, port))
        if result == 0:
            open_ports.append(port)
            print(f"  [+] Port {port}: {GREEN}OPEN{RESET}")
        s.close()
        time.sleep(0.05) # Tiny delay to simulate stealth pacing

    if not open_ports:
        print(f"  [-] No open ports discovered on {target_ip}.")
    else:
        print(f"{GREEN}[+] Scan Complete. Discovered {len(open_ports)} open ports.{RESET}")

async def main():
    parser = argparse.ArgumentParser(description="DeviceDNA Attacker CLI (Hardware PRD Mode)")
    parser.add_argument("--target", help="Exploit target IP address")
    parser.add_argument("--device-id", help="Exploit target device ID")
    parser.add_argument("--attacker-ip", help="Attacker machine local IP address")
    parser.add_argument("--attack", choices=["recon", "beacon", "lateral", "exfil", "coordinated"], help="Threat scenario to trigger")
    args = parser.parse_args()

    # Automatic / Non-interactive CLI invocation
    if args.attack:
        if args.attack == "recon":
            run_nmap_scan(args.target)
        else:
            if not args.attacker_ip or not args.device_id or not args.target:
                print("Error: --attacker-ip, --device-id, and --target parameters are required.")
                return
            if args.attack in ["beacon", "coordinated"]:
                start_c2_listener()
            if args.attack in ["exfil", "coordinated"]:
                start_exfil_listener()
            await trigger_exploit(args.target, args.device_id, args.attacker_ip, args.attack)
        return

    # Interactive Wizard Mode
    while True:
        print(f"\n{BOLD}{MAGENTA}================================================================{RESET}")
        print(f"{BOLD}{MAGENTA}                DEVICEDNA - ATTACKER LAPTOP CLI                 {RESET}")
        print(f"{BOLD}{MAGENTA}================================================================{RESET}")
        print("  1. View Fleet Status (Live API)")
        print("  2. Launch Threat Scenario")
        print("  3. Exit")
        
        choice = input(f"\n{BOLD}Select an option [1-3]: {RESET}").strip()
        
        if choice == '1':
            fetch_active_fleet()
        elif choice == '2':
            devices = fetch_active_fleet()
            if not devices:
                continue

            target_idx = input(f"\n{BOLD}Enter Device ID or Target IP Address to attack: {RESET}").strip()
            # Match input to target device IP
            device = next((d for d in devices if d["id"] == target_idx or d["ip_address"] == target_idx), None)
            if not device:
                print(f"{RED}[x] Device {target_idx} not found in current fleet config.{RESET}")
                continue

            print(f"\n{BOLD}{CYAN}Choose Attack Scenario:{RESET}")
            print("  1. Attack 1: Stealth TCP Recon Scan (Local TCP Port Sweep)")
            print("  2. Attack 2: Botnet C2 Beaconing (Outbound beacons back to attacker)")
            print("  3. Attack 3: Lateral Movement Probes (Simulate cross-node SSH/Modbus/MQTT connections)")
            print("  4. Attack 4: Slow Data Exfiltration (Incremental data dumps to attacker exfil port)")
            print("  5. Attack 5: Coordinated Botnet DDoS (Beacon + Lateral + Exfil concurrent execution)")
            
            attack_choice = input(f"\n{BOLD}Select attack type [1-5]: {RESET}").strip()
            
            if attack_choice == '1':
                run_nmap_scan(device["ip_address"])
            elif attack_choice in ['2', '3', '4', '5']:
                attacker_ip = input(f"{BOLD}Enter Attacker Laptop IP Address: {RESET}").strip()
                if not attacker_ip:
                    print(f"{RED}[x] Attacker IP is required.{RESET}")
                    continue

                # Start listeners based on attack selection
                if attack_choice in ['2', '5']:
                    start_c2_listener()
                if attack_choice in ['4', '5']:
                    start_exfil_listener()

                attack_type_map = {
                    '2': 'beacon',
                    '3': 'lateral',
                    '4': 'exfil',
                    '5': 'coordinated'
                }
                
                # Send the exploit trigger packet to the simulator's host IP (port 8888)
                # In Docker Compose setup, the simulator host target can be localhost or backend container IP
                simulator_host = input(f"{BOLD}Enter Simulator Host IP [default: localhost]: {RESET}").strip() or "localhost"
                await trigger_exploit(simulator_host, device["id"], attacker_ip, attack_type_map[attack_choice])
                
                # Give background listeners a second to show message logs
                await asyncio.sleep(2.0)
        elif choice == '3':
            print(f"\n{GREEN}[+] Stopping listeners and exiting. Attacker laptop session terminated.{RESET}")
            global C2_SERVER_ACTIVE, EXFIL_SERVER_ACTIVE
            C2_SERVER_ACTIVE = False
            EXFIL_SERVER_ACTIVE = False
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
