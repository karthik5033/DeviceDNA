import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacker_laptop.attack_orchestrator import run_nmap_scan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack 1 - Stealth Reconnaissance (Socket Sweep)")
    parser.add_argument("target", help="Target IP address to scan")
    args = parser.parse_args()
    
    run_nmap_scan(args.target)
