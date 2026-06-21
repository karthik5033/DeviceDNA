import argparse
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacker_laptop.attack_orchestrator import trigger_exploit

async def main():
    parser = argparse.ArgumentParser(description="Attack 3 - Lateral Movement Trigger")
    parser.add_argument("--target", default="localhost", help="Target simulator host (default: localhost)")
    parser.add_argument("--device-id", required=True, help="Target device ID to compromise (e.g. cam_01)")
    parser.add_argument("--attacker-ip", required=True, help="Local IP of the attacker machine")
    args = parser.parse_args()

    # Trigger the lateral exploit on the target device
    await trigger_exploit(args.target, args.device_id, args.attacker_ip, "lateral")
    
    print("[*] Lateral movement scenario triggered. The device will probe peer nodes in the background.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
