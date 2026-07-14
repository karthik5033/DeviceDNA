# Backend Code Quality & Logic Audit (Person 1 & Person 2 Modules)

## Person 1: Risk, Policy, Decay, and Recovery

### 1. Risk Classifier (`trust_engine.py`)
- **BUG (KeyError Risk)**: Line 287 reads `prev_score = float(prev_data.get("raw_score", prev_data["score"]))`. If the Redis JSON payload lacks the `"score"` key, `prev_data["score"]` will raise a `KeyError` before the `.get` default evaluation can occur. It should be `prev_data.get("raw_score", prev_data.get("score"))`.
- **LOGIC GAP (Peer Comparison compounding)**: `peer_score = 1.0 - abs(preliminary_trust_score - class_mean) / 100.0`. `abs()` penalizes a device for deviating from the mean in *either* direction. If the `class_mean` drops to 60 because multiple devices are compromised, a perfectly healthy device (score 100) will be penalized by 40 points simply for not acting compromised. 
- **STATE MANAGEMENT (Active Attacks)**: `self.active_attacks` is an in-memory dictionary. In a multi-worker environment (like Uvicorn with workers > 1) or upon a pod restart, this state is lost or fragmented. It should use Redis like the rest of the application.
- **HARDCODED VALUES**: `CLASS_POLICY_RULES` hardcodes numeric feature vector indices (e.g., `13`, `1`, `11`) and explicit device classes (`camera`, `thermostat`, etc.), violating decoupling principles.

### 2. Policy Engine (`policy_parser.py` & `trust_engine.py`)
- **BUG (Feature Name Mismatches)**: `policy_parser.py` generates natural language mapping strings that are physically impossible to match against `schemas/features.py`. 
  - Generates `bytes_sent > 5242880`, but the feature schema expects `total_bytes`.
  - Generates `ext_int_ratio > 0.8`, but the schema expects `external_traffic_ratio`.
- **BUG (Unparseable Operators)**: `policy_parser.py` generates conditions like `new_port_detected NOT IN device.dna.ports` and `dst_ip IN threat_feed.tor_exits`. The regex parser in `trust_engine.py`'s `evaluate_policy_score` (`re.match(r"(\w+)\s*([<>=!]+)\s*([\d\.\-]+)", condition_str)`) does not support the `IN` or `NOT IN` operators, causing these critical policies to be silently bypassed as "passed".

### 3. Human Override (`response.py` & `response_engine.py`)
- **LOGIC GAP (Override Blindness)**: When a HITL action is denied, an override key is set for 5 minutes (`response:override:{device_id}`). However, `response_engine.py` skips *all* automated evaluations while this key exists. If an operator denies a Tier 4 (Quarantine) action, and the device subsequently drops to a Tier 5 (Honeypot) threat level, the engine will blindly ignore it for 5 minutes.

### 4. Recovery Manager (`recovery_manager.py`)
- **LOGIC GAP (Overwriting Manual Actions)**: The Recovery Manager clears all restriction keys (`rate_limit`, `sandboxed`, `isolated`, `honeypot`) automatically when `multiplier >= 1.0`. There is no mechanism to differentiate between an automated sandbox and a *manual* sandbox applied by an operator. The Recovery Manager will silently undo an operator's manual isolation after 5 minutes of "clean" behavior.

---

## Person 2: MQTT, Responses, Sandbox, and Audit

### 5. MQTT Dispatcher (`mqtt_dispatcher.py`)
- **BUG (Asyncio Loop Reference)**: `MqttCommandDispatcher` is instantiated globally at the bottom of the file. Its `__init__` tries to set up reconnect logic, but `asyncio.get_running_loop()` will be `None` at import time. The reconnect task uses `call_soon_threadsafe(self._create_reconnect_task)`, which can fail or lose reference if the application loop isn't securely captured.
- **HARDCODED CONFIG**: `client_id="devicedna_backend_dispatcher"` is globally hardcoded. If the backend scales horizontally (multiple instances), they will constantly boot each other off the MQTT broker due to identical client IDs (connection flapping).

### 6. Response Engine (`response_engine.py`)
- **LOGIC GAP (State Stacking)**: When a device's trust score improves (e.g., from 30 back up to 50), `evaluate_triggers` applies the Tier 3 `sandbox_device`. However, it never explicitly removes the Tier 4 `isolated` state. A device can become simultaneously rate-limited, sandboxed, and quarantined in Redis. Transitions between tiers should explicitly clear mutually exclusive states.
- **HARDCODED TTLs**: Expiration timers for actions (`ISOLATION_TTL = 3600`, etc.) are hardcoded at the top of the file, completely ignoring the dynamic `time_constraint` values that the Policy Engine might parse (e.g., `window=1h`).

### 7. Audit Logs (`audit.py`)
- **BUG (Malformed Timestamps)**: Exactly like the bug previously found in `alerts.py`, `_serialize()` forcibly appends `"Z"` to `log.timestamp.isoformat()`. Because the SQLAlchemy column uses `timezone.utc`, `.isoformat()` already outputs `+00:00`. The result is a malformed double-timezone string (e.g., `2026-06-20T08:00:00+00:00Z`) which will crash strict ISO-8601 frontend parsers.
