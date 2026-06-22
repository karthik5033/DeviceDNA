import time
import json
import redis
import argparse

def main():
    parser = argparse.ArgumentParser(description="Attack 4 - Data Exfiltration")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--duration", type=int, default=300, help="Attack duration in seconds")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    targets = ['SIM-0040', 'SIM-0041', 'SIM-0045']

    print('=' * 60)
    print('  ATTACK 4 -- Data Exfiltration (Ransomware / Spyware)')
    print('=' * 60)
    print(f'[ATTACK 4] Targets: {", ".join(targets)}')
    
    payload = json.dumps({"type": "exfil", "intensity": 1.0})

    for device_id in targets:
        r.set(f"attack_state:{device_id}", payload)
        print(f'  [+] {device_id} -- Injecting massive data exfiltration over HTTPS to external IP')

    print(f'\n[ATTACK 4] Attack active for {args.duration} seconds...')
    
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print('\n[ATTACK 4] Interrupted.')

    print('\n[ATTACK 4] Cleaning up...')
    for device_id in targets:
        r.delete(f"attack_state:{device_id}")
    print('[ATTACK 4] Complete.')

if __name__ == "__main__":
    main()
