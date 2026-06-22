import time
import json
import redis
import argparse

def main():
    parser = argparse.ArgumentParser(description="Attack 3 - Lateral Movement")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--duration", type=int, default=300, help="Attack duration in seconds")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    targets = ['SIM-0010', 'SIM-0011', 'SIM-0016']

    print('=' * 60)
    print('  ATTACK 3 -- Lateral Movement / Worm Spread')
    print('=' * 60)
    print(f'[ATTACK 3] Targets: {", ".join(targets)}')
    
    payload = json.dumps({"type": "lateral", "intensity": 0.8})

    for device_id in targets:
        r.set(f"attack_state:{device_id}", payload)
        print(f'  [+] {device_id} -- Injecting lateral movement behavior (scanning internal IPs on RDP/SMB/SSH)')

    print(f'\n[ATTACK 3] Attack active for {args.duration} seconds...')
    
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print('\n[ATTACK 3] Interrupted.')

    print('\n[ATTACK 3] Cleaning up...')
    for device_id in targets:
        r.delete(f"attack_state:{device_id}")
    print('[ATTACK 3] Complete.')

if __name__ == "__main__":
    main()
