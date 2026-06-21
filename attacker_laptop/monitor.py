import time
import requests
import socketio
import os
import sys
import logging
from tabulate import tabulate

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

# Constants
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "http://localhost:8000")

# Suppress noisy logging
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

class DeviceDNAMonitor:
    def __init__(self):
        self.devices = {}
        self.alerts = []
        self.responses = {}
        self.sio = socketio.Client()
        self.running = True

    def fetch_initial_state(self):
        print("Polling backend for initial platform state...")
        # 1. Fetch all devices
        try:
            r = requests.get(f"{API_BASE}/api/trust/devices/all", timeout=5)
            if r.status_code == 200:
                for dev in r.json():
                    self.devices[dev["id"]] = {
                        "name": dev["name"],
                        "class": dev["device_class"],
                        "ip": dev["ip_address"],
                        "score": dev["trust_score"],
                        "status": dev["status"]
                    }
            else:
                print(f"Warning: Failed to fetch devices: {r.status_code}")
        except Exception as e:
            print(f"Error fetching initial device state: {e}")

        # 2. Fetch initial alerts
        try:
            r = requests.get(f"{API_BASE}/api/alerts", timeout=5)
            if r.status_code == 200:
                self.alerts = r.json()[:10]  # Keep last 10 alerts
            else:
                print(f"Warning: Failed to fetch alerts: {r.status_code}")
        except Exception as e:
            print(f"Error fetching initial alerts: {e}")

        # 3. Fetch response status for compromised/low score nodes
        try:
            for dev_id in self.devices.keys():
                # Only check if trust score is below 95 to avoid excessive spamming
                if self.devices[dev_id]["score"] < 95.0:
                    r = requests.get(f"{API_BASE}/api/response/{dev_id}/status", timeout=2)
                    if r.status_code == 200:
                        self.responses[dev_id] = r.json()
        except Exception as e:
            print(f"Error fetching response states: {e}")

    def setup_websockets(self):
        @self.sio.on('connect')
        def on_connect():
            pass

        @self.sio.on('trust_update')
        def on_trust_update(data):
            # Format: {'device_id': '...', 'score': 85.0, 'timestamp': '...'}
            dev_id = data.get("device_id")
            if dev_id in self.devices:
                self.devices[dev_id]["score"] = data.get("score", 100.0)
                score = data.get("score", 100.0)
                if score >= 80:
                    self.devices[dev_id]["status"] = "trusted"
                elif score >= 60:
                    self.devices[dev_id]["status"] = "guarded"
                elif score >= 40:
                    self.devices[dev_id]["status"] = "suspicious"
                else:
                    self.devices[dev_id]["status"] = "critical"
            
            # Fetch updated response status for this device
            try:
                r = requests.get(f"{API_BASE}/api/response/{dev_id}/status", timeout=1)
                if r.status_code == 200:
                    self.responses[dev_id] = r.json()
            except Exception:
                pass

        @self.sio.on('new_alert')
        def on_new_alert(data):
            # Add to the top of alerts list
            self.alerts.insert(0, data)
            if len(self.alerts) > 10:
                self.alerts.pop()
            
            # Fetch updated response status for the alerted device
            dev_id = data.get("device")
            if dev_id:
                try:
                    r = requests.get(f"{API_BASE}/api/response/{dev_id}/status", timeout=1)
                    if r.status_code == 200:
                        self.responses[dev_id] = r.json()
                except Exception:
                    pass

        @self.sio.on('hitl_pending')
        @self.sio.on('rate_limit_device')
        @self.sio.on('sandbox_device')
        @self.sio.on('isolate_device')
        @self.sio.on('honeypot_device')
        def on_response_action(data):
            dev_id = data.get("device_id") or data.get("device")
            if dev_id:
                try:
                    r = requests.get(f"{API_BASE}/api/response/{dev_id}/status", timeout=1)
                    if r.status_code == 200:
                        self.responses[dev_id] = r.json()
                except Exception:
                    pass

    def run(self):
        self.fetch_initial_state()
        self.setup_websockets()
        
        try:
            self.sio.connect(WS_URL)
        except Exception as e:
            print(f"Failed to connect to backend WebSockets: {e}. Running in HTTP-only polling mode.")

        # Main print loop
        try:
            while self.running:
                # Clear terminal screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("==========================================================================")
                print("                     DNA DeviceDNA - LIVE THREAT MONITOR                  ")
                print("==========================================================================")

                # 1. Summary Metrics
                total_devices = len(self.devices)
                avg_trust = sum(d["score"] for d in self.devices.values()) / total_devices if total_devices > 0 else 100.0
                suspicious_count = sum(1 for d in self.devices.values() if d["score"] < 80.0)
                critical_count = sum(1 for d in self.devices.values() if d["score"] < 40.0)
                
                active_quarantines = 0
                active_rate_limits = 0
                for resp in self.responses.values():
                    if resp.get("isolated"):
                        active_quarantines += 1
                    if resp.get("rate_limited"):
                        active_rate_limits += 1

                print(f"Total Fleet Devices: {total_devices} | Average Trust Score: {avg_trust:.2f}")
                print(f"Threat Levels:       {suspicious_count} Suspicious | {critical_count} Critical")
                print(f"Active Defenses:     {active_rate_limits} Rate Limited | {active_quarantines} Quarantined")
                print("--------------------------------------------------------------------------")

                # 2. Compromised / Scored Devices (Trust Score < 95)
                compromised_list = []
                for dev_id, dev in self.devices.items():
                    if dev["score"] < 98.0: # Show devices showing any signs of deviation
                        resp = self.responses.get(dev_id, {})
                        active_def = []
                        if resp.get("rate_limited"): active_def.append("Rate-Limit")
                        if resp.get("sandboxed"): active_def.append("Sandbox")
                        if resp.get("isolated"): active_def.append("Quarantine")
                        if resp.get("honeypot"): active_def.append("Honeypot")
                        if resp.get("pending_approval"): active_def.append("PENDING-HITL")
                        
                        def_status = ", ".join(active_def) if active_def else "None"
                        compromised_list.append([
                            dev_id,
                            dev["name"],
                            dev["class"],
                            dev["ip"],
                            f"{dev['score']:.2f}",
                            dev["status"].upper(),
                            def_status
                        ])
                
                print("!!! ANOMALOUS & COMPROMISED DEVICES:")
                if compromised_list:
                    print(tabulate(
                        compromised_list,
                        headers=["Device ID", "Name", "Class", "IP Address", "Trust Score", "Status", "Active Defenses"],
                        tablefmt="simple"
                    ))
                else:
                    print("  [Green] All device trust scores at baseline (100.0). No anomalies detected.")
                print("--------------------------------------------------------------------------")

                # 3. Live Alert Feed
                print("[*] LIVE SECURITY ALERT FEED (Last 5):")
                if self.alerts:
                    for idx, alert in enumerate(self.alerts[:5]):
                        # Handle either REST schema (created_at) or WS schema (time)
                        time_val = alert.get("time") or alert.get("timestamp")
                        # Truncate timestamp for display
                        if time_val and len(time_val) > 19:
                            time_val = time_val[11:19]
                        
                        severity = alert.get("severity", "medium").upper()
                        print(f"  [{time_val}] [{severity}] {alert.get('device', 'Fleet')}: {alert.get('message')}")
                else:
                    print("  No alerts triggered in this session.")
                print("==========================================================================")
                print("Press Ctrl+C to stop monitor.")
                
                time.sleep(2)
        except KeyboardInterrupt:
            self.running = False
        finally:
            if self.sio.connected:
                self.sio.disconnect()
            print("\nMonitor terminated.")

if __name__ == "__main__":
    monitor = DeviceDNAMonitor()
    monitor.run()
