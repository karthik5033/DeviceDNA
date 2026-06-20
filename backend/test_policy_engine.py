import sys
from unittest.mock import MagicMock

# Mock InfluxDBClientAsync to prevent event loop errors during import time
class MockInfluxClient:
    def __init__(self, *args, **kwargs):
        pass
    def write_api(self, *args, **kwargs):
        return MagicMock()
    def query_api(self, *args, **kwargs):
        return MagicMock()
    async def close(self):
        pass

mock_module = MagicMock()
mock_module.InfluxDBClientAsync = MockInfluxClient
sys.modules['influxdb_client.client.influxdb_client_async'] = mock_module


import asyncio
import unittest
from sqlalchemy import select

# Add parent path to import app correctly
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.postgres import AsyncSessionLocal, engine, Base
from app.db.models import PolicyRule
from app.ml.nlp.policy_parser import parse_policy
from app.services.trust_engine import evaluate_policy_score



class TestPolicyEngine(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Clear connection pool to start on the correct loop
        await engine.dispose()
        # Clean up existing rules to start fresh
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(PolicyRule))
            rules = result.scalars().all()
            for rule in rules:
                await session.delete(rule)
            await session.commit()

    async def asyncTearDown(self):
        # Clear connection pool after test
        await engine.dispose()

    async def test_1_regex_fallback_parsing(self):
        """Test regex fallback parser matches core scenarios correctly."""
        print("\nRunning test_1_regex_fallback_parsing...")
        
        # Test Case A: Camera external IP after midnight
        res1 = parse_policy("alert if any camera contacts external IP after midnight")
        self.assertEqual(res1["intent"], "alert")
        self.assertEqual(res1["device_class"], "camera")
        self.assertEqual(res1["condition"], "ext_int_ratio > 0.8")
        self.assertEqual(res1["time_constraint"], "hour >= 0 AND hour < 6")
        
        # Test Case B: Isolate device opening new port
        res2 = parse_policy("isolate any device that opens a new port not in its baseline")
        self.assertEqual(res2["intent"], "isolate")
        self.assertEqual(res2["device_class"], "any")
        self.assertEqual(res2["condition"], "new_port_detected NOT IN device.dna.ports")
        self.assertEqual(res2["severity"], "CRITICAL")
        
        # Test Case C: Upload limit
        res3 = parse_policy("alert if upload exceeds 5 MB in one hour")
        self.assertEqual(res3["intent"], "alert")
        self.assertEqual(res3["condition"], "bytes_sent > 5242880")
        self.assertEqual(res3["time_constraint"], "window=1h")

    async def test_2_dynamic_policy_evaluation(self):
        """Test dynamic policy database rules override and evaluate correctly."""
        print("Running test_2_dynamic_policy_evaluation...")
        
        # Scenario: A Camera with normal features
        # Indices: 8 is ext_int_ratio, 0 is bytes_sent
        # Normal feature vector: ext_int_ratio = 0.2, bytes_sent = 5000 (compliant with static rules)
        features = [5000.0] + [0.0]*7 + [0.2] + [0.0]*5
        
        # 1. No custom active DB rules -> should evaluate to 1.0 (all static rules pass)
        score_before = await evaluate_policy_score("camera", features)
        self.assertEqual(score_before, 1.0)
        
        # 2. Add an active custom DB rule restricting ext_int_ratio to 0.1
        async with AsyncSessionLocal() as session:
            rule = PolicyRule(
                device_class="camera",
                condition="ext_int_ratio > 0.1", # Violates since our ext_int_ratio is 0.2
                action="alert",
                severity="HIGH",
                natural_language_rule="Alert if camera ext_int_ratio is high",
                is_active=True
            )
            session.add(rule)
            await session.commit()
            
        # 3. Re-evaluate. With 1 active DB rule violated, score should drop to 0.0
        score_after = await evaluate_policy_score("camera", features)
        self.assertEqual(score_after, 0.0)

    async def test_3_policy_toggle_and_deactivation(self):
        """Test that deactivating/toggling a rule returns the score back to passing."""
        print("Running test_3_policy_toggle_and_deactivation...")
        
        # 1. Add rule violating: bytes_sent > 1000
        features = [5000.0] + [0.0]*13 # bytes_sent = 5000 (violates rule)
        
        async with AsyncSessionLocal() as session:
            rule = PolicyRule(
                device_class="any",
                condition="bytes_sent > 1000",
                action="alert",
                is_active=True
            )
            session.add(rule)
            await session.commit()
            rule_id = rule.id
            
        score_active = await evaluate_policy_score("sensor", features)
        self.assertEqual(score_active, 0.0) # Violated
        
        # 2. Deactivate the rule
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(PolicyRule).filter(PolicyRule.id == rule_id))
            db_rule = result.scalar_one()
            db_rule.is_active = False
            await session.commit()
            
        score_inactive = await evaluate_policy_score("sensor", features)
        self.assertEqual(score_inactive, 1.0) # Back to passing fallback rules

if __name__ == "__main__":
    unittest.main()
