"""
Test script to verify the 'Rate of Decline' logic in ResponseEngine.
"""
import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.response_engine import response_engine
from app.db.redis import redis_client

async def run_decline_test():
    device_id = "TEST-DECLINE"
    
    print("\n--- Testing Rate of Decline Logic ---\n")
    
    # Clean up
    for key in redis_client.keys(f"response:*:{device_id}"):
        redis_client.delete(key)
        
    print("Test 1: Slow decline (75 -> 65), Drop=10. Should just trigger Tier 2 (Rate Limit)")
    triggered = await response_engine.evaluate_triggers(
        device_id=device_id,
        trust_score=65.0,
        gnn_score=0.1,
        previous_trust_score=75.0
    )
    print(f"Triggered: {triggered}\n")
    
    # Clean up
    for key in redis_client.keys(f"response:*:{device_id}"):
        redis_client.delete(key)

    print("Test 2: Fast decline (90 -> 65), Drop=25. Should accelerate +2 tiers to Tier 4 (Quarantine)")
    triggered = await response_engine.evaluate_triggers(
        device_id=device_id,
        trust_score=65.0,
        gnn_score=0.1,
        previous_trust_score=90.0
    )
    print(f"Triggered: {triggered}\n")
    
if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_decline_test())
