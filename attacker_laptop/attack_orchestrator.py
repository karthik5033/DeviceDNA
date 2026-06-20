import argparse
import asyncio
import sys
import logging
import os
import requests
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacker_laptop.attack1_recon import run_recon_attack
from attacker_laptop.attack2_botnet import run_botnet_attack
from attacker_laptop.attack3_lateral import run_lateral_attack
from attacker_laptop.attack4_exfil import run_exfil_attack
from attacker_laptop.attack5_coordinated import run_coordinated_attack
from attacker_laptop.common.config import API_BASE_URL, FLEET

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Set logger levels to suppress verbose logs during interactive UI
logging.getLogger("AttackOrchestrator").setLevel(logging.INFO)
logging.getLogger("kafka").setLevel(logging.WARNING)

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

def fetch_active_fleet():
    """
    Fetch the list of active devices and their IP addresses from the backend API.
    If the backend is not reachable, fall back to the local simulator configurations.
    """
    print(f"\n{CYAN}[~] Fetching live fleet state from backend...{RESET}")
    try:
        r = requests.get(f"{API_BASE_URL}/api/trust/devices/all", timeout=3)
        if r.status_code == 200:
            devices = r.json()
            if devices:
                print(f"{GREEN}[+] Successfully fetched {len(devices)} active devices from backend!{RESET}")
                return devices
    except Exception as e:
        print(f"{YELLOW}[!] Warning: Could not connect to backend API: {e}{RESET}")
        print(f"{YELLOW}Using local offline FLEET definitions.{RESET}")
    
    # Fallback to local simulator configuration
    try:
        from simulator.device_profiles import FLEET as LOCAL_FLEET
        return [
            {
                "id": d["id"],
                "name": d["name"],
                "device_class": d["device_class"],
                "ip_address": d["ip_address"],
                "vlan": d["vlan"],
                "trust_score": 100.0,
                "status": "online"
            }
            for d in LOCAL_FLEET
        ]
    except Exception as err:
        print(f"{RED}[x] Error: Local FLEET definition not found: {err}. Attacker cannot run.{RESET}")
        return []

def display_fleet(devices):
    """
    Display the devices in a cleanly formatted table using tabulate.
    """
    headers = [
        f"{BOLD}Index{RESET}",
        f"{BOLD}Device ID{RESET}",
        f"{BOLD}Name{RESET}",
        f"{BOLD}Class{RESET}",
        f"{BOLD}IP Address{RESET}",
        f"{BOLD}Trust Score{RESET}",
        f"{BOLD}Status{RESET}"
    ]
    table_data = []
    for idx, d in enumerate(devices, 1):
        # Resolve score from either key name ('trust_score' or 'score')
        score = d.get("trust_score")
        if score is None:
            score = d.get("score", 100.0)
            
        status = d.get("status")
        if not status:
            if score >= 80:
                status = "trusted"
            elif score >= 60:
                status = "guarded"
            elif score >= 40:
                status = "suspicious"
            else:
                status = "critical"
        
        status = status.upper()
        
        # Color coding status
        if "CRITICAL" in status:
            status_colored = f"{RED}{status}{RESET}"
        elif "SUSPICIOUS" in status or "GUARDED" in status:
            status_colored = f"{YELLOW}{status}{RESET}"
        else:
            status_colored = f"{GREEN}{status}{RESET}"
            
        table_data.append([
            idx,
            d["id"],
            d["name"],
            d["device_class"],
            d["ip_address"],
            f"{score:.2f}",
            status_colored
        ])
        
    print("\n" + "=" * 80)
    print(f"                     [DeviceDNA Fleet Status]                     ")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print("=" * 80 + "\n")

def resolve_device(selection, devices):
    """
    Resolves user input (index, ID, or IP) to a device dictionary from the fleet.
    """
    if not selection:
        return None
    # 1. Try resolving by Index (1-based)
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(devices):
            return devices[idx]
    except ValueError:
        pass
        
    # 2. Try resolving by Device ID (case insensitive)
    dev = next((d for d in devices if d["id"].lower() == selection.lower()), None)
    if dev:
        return dev
        
    # 3. Try resolving by IP Address
    dev = next((d for d in devices if d["ip_address"] == selection), None)
    if dev:
        return dev
        
    return None

async def launch_attack(attack_func, *args, **kwargs):
    """
    Launch the attack scenario asynchronously and handle user cancellation (Ctrl+C).
    """
    attack_task = asyncio.create_task(attack_func(*args, **kwargs))
    try:
        print("\n" + "*" * 60)
        print(f"{RED}{BOLD}ATTACK INJECTOR ACTIVE & GENERATING MALICIOUS FLOWS{RESET}")
        print(f"{YELLOW}Press Ctrl+C to terminate attack and return to menu.{RESET}")
        print("*" * 60 + "\n")
        await attack_task
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopping attack execution...{RESET}")
    finally:
        if not attack_task.done():
            attack_task.cancel()
            try:
                await attack_task
            except asyncio.CancelledError:
                pass
        print(f"{GREEN}Attack stopped. Telemetry feed returned to baseline.{RESET}\n")
        await asyncio.sleep(1.5)

async def select_and_run_attack(devices):
    """
    Render options to choose which attack to inject.
    """
    print(f"\n{BOLD}{RED}[!] Select Attack Scenario to Launch:{RESET}")
    print("-" * 60)
    print(f"1. {CYAN}Stealth Reconnaissance{RESET} (Scans multiple targets using TCP_SYN)")
    print(f"2. {CYAN}Botnet C2 Beaconing{RESET} (Periodic beaconing to anomalous port - Choose Device)")
    print(f"3. {CYAN}Lateral Movement{RESET} (Spawns cross-class internal connection paths)")
    print(f"4. {CYAN}Slow Data Exfiltration{RESET} (Slowly leaks packets/bytes size over time)")
    print(f"5. {CYAN}Coordinated Multi-device Compromise{RESET} (Clustered Botnet + Lateral)")
    print(f"6. {YELLOW}Back to Main Menu{RESET}")
    print("-" * 60)
    
    choice = input(f"{BOLD}Select option (1-6): {RESET}").strip()
    if choice == "6" or not choice:
        return
        
    target_device = None
    
    # Attacks 1, 2, 3, 4 require a single target device
    if choice in ["1", "2", "3", "4"]:
        if choice == "2":
            # Botnet gets option to choose target device from list
            print(f"\n{BOLD}Select target device to recruit into the Botnet:{RESET}")
            headers = [f"{BOLD}Index{RESET}", f"{BOLD}Device ID{RESET}", f"{BOLD}Name{RESET}", f"{BOLD}Class{RESET}", f"{BOLD}IP Address{RESET}"]
            table_data = [[idx, d["id"], d["name"], d["device_class"], d["ip_address"]] for idx, d in enumerate(devices, 1)]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))
            
            while True:
                sel = input(f"\n{BOLD}Choose target device (Enter index 1-50, Device ID, or IP): {RESET}").strip()
                if not sel:
                    print(f"{YELLOW}Botnet target selection cancelled.{RESET}")
                    return
                target_device = resolve_device(sel, devices)
                if target_device:
                    break
                print(f"{RED}Invalid selection. Try again or press Enter to cancel.{RESET}")
        else:
            # Other attacks let you type the host's IP address
            while True:
                ip_input = input(f"\n{BOLD}Enter target host IP address: {RESET}").strip()
                if not ip_input:
                    print(f"{YELLOW}Attack targeting cancelled.{RESET}")
                    return
                
                target_device = next((d for d in devices if d["ip_address"] == ip_input), None)
                if target_device:
                    print(f"{GREEN}[+] Target host IP resolved to: {target_device['id']} ({target_device['name']}){RESET}")
                    break
                
                print(f"{RED}[x] No device found with IP address: {ip_input}{RESET}")
                fallback = input("Would you like to select from the fleet list instead? (y/n): ").strip().lower()
                if fallback == "y":
                    headers = [f"{BOLD}Index{RESET}", f"{BOLD}Device ID{RESET}", f"{BOLD}Name{RESET}", f"{BOLD}Class{RESET}", f"{BOLD}IP Address{RESET}"]
                    table_data = [[idx, d["id"], d["name"], d["device_class"], d["ip_address"]] for idx, d in enumerate(devices, 1)]
                    print(tabulate(table_data, headers=headers, tablefmt="simple"))
                    
                    sel = input(f"\n{BOLD}Choose target device (Enter index 1-50, ID, or IP): {RESET}").strip()
                    target_device = resolve_device(sel, devices)
                    if target_device:
                        break
                else:
                    retry = input("Try typing another IP address? (y/n): ").strip().lower()
                    if retry != "y":
                        return

    # Trigger selected attack function
    if choice == "1":
        await launch_attack(run_recon_attack, target_id=target_device["id"])
    elif choice == "2":
        await launch_attack(run_botnet_attack, target_id=target_device["id"])
    elif choice == "3":
        await launch_attack(run_lateral_attack, target_id=target_device["id"])
    elif choice == "4":
        await launch_attack(run_exfil_attack, target_id=target_device["id"])
    elif choice == "5":
        confirm = input(f"{BOLD}Launch Coordinated Attack on [SIM-0001, SIM-0007, SIM-0021, SIM-0032]? (y/n): {RESET}").strip().lower()
        if confirm == "y":
            await launch_attack(run_coordinated_attack)
    else:
        print(f"{RED}Invalid attack selection.{RESET}")

async def run_interactive_menu():
    """
    Main interactive loop for the Attacker CLI application.
    """
    devices = fetch_active_fleet()
    if not devices:
        print(f"{RED}Cannot start CLI Application: Fleet definition not available.{RESET}")
        return
        
    # Patch FLEET list in-place so submodules use the correct dynamic IPs
    FLEET.clear()
    FLEET.extend(devices)

    while True:
        try:
            print(f"\n{BOLD}[DeviceDNA Attacker Laptop — Interactive CLI Control]{RESET}")
            print("-" * 60)
            print(f"1. {CYAN}View Active Device Fleet{RESET}")
            print(f"2. {RED}Launch Threat/Attack Scenario{RESET}")
            print(f"3. {YELLOW}Exit{RESET}")
            print("-" * 60)
            
            choice = input(f"{BOLD}Select option (1-3): {RESET}").strip()
            
            if choice == "1":
                # Refresh fleet before showing
                refreshed = fetch_active_fleet()
                if refreshed:
                    devices = refreshed
                    FLEET.clear()
                    FLEET.extend(devices)
                display_fleet(devices)
                input(f"\n{BOLD}Press Enter to return to main menu...{RESET}")
            elif choice == "2":
                await select_and_run_attack(devices)
            elif choice == "3":
                print(f"\n{GREEN}Goodbye! Attacker shutting down.{RESET}")
                break
            else:
                print(f"{RED}Invalid option. Please choose 1, 2, or 3.{RESET}")
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}To exit the application, please use option 3.{RESET}")
        except EOFError:
            break

async def main():
    parser = argparse.ArgumentParser(description="DeviceDNA Attacker Laptop Orchestrator CLI")
    parser.add_argument(
        "--attack",
        choices=["recon", "botnet", "lateral", "exfil", "coordinated"],
        help="Specify the attack scenario to run."
    )
    parser.add_argument(
        "--ip",
        help="Specify target device IP address (for argument-based run)."
    )
    parser.add_argument(
        "--device-id",
        help="Specify target device ID (for argument-based run)."
    )
    args = parser.parse_args()
    
    # If no arguments provided, launch the interactive UI menu
    if args.attack is None:
        await run_interactive_menu()
        return

    # Non-interactive argument-based execution
    devices = fetch_active_fleet()
    if not devices:
        logger.error("Failed to load fleet mapping. Aborting.")
        sys.exit(1)
        
    FLEET.clear()
    FLEET.extend(devices)
    
    # Resolve target device
    target_device = None
    if args.attack != "coordinated":
        if args.ip:
            target_device = resolve_device(args.ip, devices)
        elif args.device_id:
            target_device = resolve_device(args.device_id, devices)
            
        if not target_device:
            logger.error("A valid target device (--ip or --device-id) is required for this attack.")
            sys.exit(1)
            
        logger.info(f"Target resolved: {target_device['id']} - {target_device['name']} ({target_device['ip_address']})")

    # Launch non-interactive attack
    attack_task = None
    try:
        if args.attack == "recon":
            attack_task = asyncio.create_task(run_recon_attack(target_id=target_device["id"]))
        elif args.attack == "botnet":
            attack_task = asyncio.create_task(run_botnet_attack(target_id=target_device["id"]))
        elif args.attack == "lateral":
            attack_task = asyncio.create_task(run_lateral_attack(target_id=target_device["id"]))
        elif args.attack == "exfil":
            attack_task = asyncio.create_task(run_exfil_attack(target_id=target_device["id"]))
        elif args.attack == "coordinated":
            attack_task = asyncio.create_task(run_coordinated_attack())
            
        logger.info(f"Attack scenario '{args.attack}' is running. Press Ctrl+C to terminate.")
        await attack_task
    except KeyboardInterrupt:
        logger.info("Termination requested by user (Ctrl+C). Cleaning up...")
        if attack_task and not attack_task.done():
            attack_task.cancel()
            try:
                await attack_task
            except asyncio.CancelledError:
                pass
        logger.info("Attack successfully terminated.")
    except Exception as e:
        logger.error(f"Error during attack execution: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
