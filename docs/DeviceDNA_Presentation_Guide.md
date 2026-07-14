# DeviceDNA — IoT Threat Detection & Autonomous Response Platform
### Comprehensive Project Documentation & Viva Preparation Guide

---

## 1. Why DeviceDNA? — The Problem Statement

### The IoT Security Crisis
By 2025, there are over **15 billion IoT devices** globally — smart cameras, medical sensors, thermostats, industrial controllers — and the number is growing at 20% per year. These devices have a critical flaw: **they are resource-constrained** and cannot run traditional antivirus, firewall, or endpoint detection software.

### Key Vulnerabilities
| Problem | Impact |
|---|---|
| No antivirus/EDR on IoT devices | Attacks go undetected for months |
| Default credentials | 50%+ of IoT devices ship with `admin/admin` |
| No automatic patching | Firmware vulnerabilities remain exploitable |
| Flat network architectures | One compromised camera can reach the database server |
| No behavioral monitoring | Traditional SIEM/IDS tools ignore IoT-specific patterns |

### Real-World Attack Examples
- **Mirai Botnet (2016)**: Infected 600,000+ IoT cameras/routers, launched 1.2 Tbps DDoS attack on Dyn DNS, taking down Twitter, Netflix, and Reddit
- **Stuxnet**: Targeted industrial IoT controllers (PLCs) in Iran's nuclear program
- **Hospital IoT Attacks**: Compromised infusion pumps and patient monitors leading to patient safety risks

### What Existing Solutions Lack
Traditional security tools (firewalls, IDS/IPS like Snort) use **signature-based detection** — they only catch known attack patterns. IoT threats are often **zero-day** or **low-and-slow**, where the attacker slowly exfiltrates data over weeks, staying below static thresholds.

> **DeviceDNA's Answer**: Use **machine learning** to learn what "normal" looks like for *each individual device*, then detect *any* deviation — even novel, never-before-seen attacks.

---

## 2. Our Approach — What Makes DeviceDNA Different

### Core Philosophy: "Behavioral DNA Fingerprinting"
Every IoT device has a **unique behavioral signature** — like a fingerprint or DNA. A temperature sensor sends ~500 bytes every 30 seconds to 2 internal IPs using MQTT. A security camera streams ~1.2MB/s to a cloud NVR via RTSP. If a sensor suddenly starts sending 5MB to an unknown external IP, *something is wrong*.

DeviceDNA:
1. **Learns** each device's normal behavioral DNA during a baseline period
2. **Monitors** live network telemetry in real-time
3. **Detects** deviations using a 5-pillar ML ensemble
4. **Responds** autonomously with a 5-tier graduated response system
5. **Explains** every decision with human-readable Threat Intelligence Briefs (TIBs)

### What It Solves
- Zero-day attack detection (no signatures needed)
- Insider threats and compromised devices
- Slow data exfiltration over days/weeks
- Lateral movement across the network
- Botnet Command and Control (C2) beaconing
- Port scanning and reconnaissance
- Autonomous response without human delay
- Physical hardware isolation via ESP32 relay actuators

---

## 3. System Architecture

### High-Level Architecture Diagram

```
+-----------------------------------------------------------------------+
|                         DeviceDNA Architecture                         |
+-----------------------------------------------------------------------+
|                                                                         |
|   +==============+    MQTT     +==================+    Kafka            |
|   |  ESP32 Nodes |----------->|  Hardware Gateway  |---------->         |
|   |  (Physical)  |  Port 1883 |  (MQTT -> Kafka)  |                    |
|   +==============+            +==================+                     |
|                                                        |               |
|   +==============+    Kafka   +============================+           |
|   |  Simulator   |---------->|        Kafka Broker         |           |
|   |  (50 Virtual |  raw-flows|    Topic: "raw-flows"       |           |
|   |   Devices)   |           +=============|==============+           |
|   +==============+                         |                          |
|                                            v                          |
|                            +============================+             |
|                            |     FastAPI Backend         |             |
|                            |  +----------------------+  |             |
|                            |  | Telemetry Consumer   |  |             |
|                            |  | Feature Extractor    |  |             |
|                            |  | Trust Engine         |  |             |
|                            |  | Response Engine      |  |             |
|                            |  | ML Scoring Suite     |  |             |
|                            |  +----------------------+  |             |
|                            +=========|==|==|============+             |
|                                      |  |  |                         |
|                     +----------------+  |  +---------------+         |
|                     v                   v                  v          |
|              +-----------+   +--------------+  +-----------+         |
|              | PostgreSQL|   |   InfluxDB    |  |   Redis   |         |
|              | (Devices, |   | (Time-Series  |  | (Real-time|         |
|              |  Alerts,  |   |  Telemetry,   |  |  Scores,  |         |
|              |  Audit)   |   |  Trust Scores)|  |  State)   |         |
|              +-----------+   +--------------+  +-----------+         |
|                                                                       |
|                            +========================+                 |
|                            |   Next.js Frontend     |                 |
|                Socket.IO   |   (Real-Time Dashboard)|                 |
|              <-------------|   Port 3000            |                 |
|                            +========================+                 |
|                                                                       |
|   +==============+  MQTT Commands  +===================+             |
|   |  ESP32 Relay |<----------------|  MQTT Dispatcher  |             |
|   |  (Actuator)  |  quarantine/    |  (Backend->ESP32) |             |
|   |  GPIO 26     |  rate_limit/    +===================+             |
|   +==============+  recover                                          |
+-----------------------------------------------------------------------+
```

### Component Breakdown

| Component | Technology | Port | Purpose |
|---|---|---|---|
| Backend API | FastAPI (Python) | 8000 | Core brain — ML inference, trust scoring, response orchestration |
| Frontend Dashboard | Next.js 14 (React) | 3000 | Real-time visualization with Socket.IO |
| PostgreSQL | v16 Alpine | 5432 | Persistent storage — devices, alerts, audit logs, policy rules |
| InfluxDB | v2.7 | 8086 | Time-series database for telemetry and trust score history |
| Redis | v7 Alpine | 6379 | In-memory cache — real-time trust scores, attack state, HITL queue |
| Apache Kafka | Confluent 7.5 | 9092 | Stream processing — ingests raw network flows at high throughput |
| Zookeeper | Confluent 7.5 | 2181 | Kafka cluster coordination |
| Mosquitto | Eclipse 2.0 | 1883 | MQTT broker — ESP32 devices publish/subscribe here |
| Hardware Gateway | Python (aiomqtt) | — | Bridge: validates MQTT payloads then forwards to Kafka |
| Simulator | Python | 8888 | Generates realistic telemetry for 50 virtual IoT devices |
| ESP32 Firmware | Arduino C++ | — | Physical relay actuator + telemetry publisher |

---

## 4. The Trust Score — How It Works

### Multi-Dimensional Trust Scoring (0-100)

The trust score is a **weighted composite** of 5 independent "pillars," each measuring a different dimension of device behavior:

```
Trust Score = 100 - (Weighted Penalty x 100)

Weighted Penalty =
    (VAE Deviation     x 0.35)   <-- Digital Twin Pillar
  + (Ensemble Score    x 0.25)   <-- Anomaly Ensemble Pillar  
  + (CUSUM Drift       x 0.20)   <-- Drift Intelligence Pillar
  + (Policy Penalty    x 0.15)   <-- Policy Conformance Pillar
  + (Peer Penalty      x 0.05)   <-- Peer Comparison Pillar
```

### The 5 Pillars Explained

#### Pillar 1: Digital Twin (35% weight) — VAE + GMVAE
- **What**: A Variational Autoencoder trained *per device* learns to reconstruct its normal 14-dimensional behavior vector
- **How**: If the reconstruction error is high, the device is behaving abnormally
- **Think of it as**: "Can I recreate this behavior from the device's learned personality?"

#### Pillar 2: Anomaly Ensemble (25% weight) — IF + LSTM + GNN
A weighted sub-ensemble of three models:
- **Isolation Forest (60%)**: Trained *per device class* (camera, sensor, etc.). Isolates outliers in feature space
- **LSTM (20%)**: Looks at the *sequence* of the last 12 feature snapshots. Catches temporal patterns (e.g., periodic beaconing)
- **GNN / GraphSAGE (20%)**: Builds a live *communication graph* between devices. Detects abnormal topology changes (e.g., a sensor suddenly talking to 10 new peers)

#### Pillar 3: Drift Intelligence (20% weight) — CUSUM
- **What**: Cumulative Sum (CUSUM) algorithm tracks slow, sustained increases in `total_bytes`, `avg_packet_size`, and `external_traffic_ratio`
- **Why**: VAE and IF catch sudden spikes, but CUSUM catches *gradual exfiltration* over hours/days
- **Math**: Z-score normalization -> cumulative positive/negative sums -> alarm when sum exceeds threshold H = 3.0

#### Pillar 4: Policy Conformance (15% weight)
- **What**: Hard-coded or NLP-defined rules per device class (e.g., "Sensors must not send > 50KB" or "Cameras must have < 60% external traffic")
- **How**: Fraction of rules passed -> penalty = 1 - pass_ratio

#### Pillar 5: Peer Comparison (5% weight)
- **What**: Compares the device's preliminary trust score against the average of all other devices of the same class
- **Why**: If all 10 sensors score 95+ but one scores 62, that outlier deserves extra scrutiny

### Trust Score Status Mapping

| Score Range | Status | Color | Meaning |
|---|---|---|---|
| 80-100 | Trusted | Green | Normal operation |
| 60-79 | Guarded | Yellow | Minor anomaly detected |
| 40-59 | Suspicious | Orange | Significant deviation |
| 0-39 | Critical | Red | Likely compromised |

### Exponential Moving Average (EMA) Smoothing
To prevent jitter, the raw score is smoothed using asymmetric EMA:
- **Score dropping** (attack detected): alpha = 0.6 -> responds fast (60% new, 40% old)
- **Score recovering** (back to normal): alpha = 0.1 -> recovers slowly (10% new, 90% old)

This means: **attacks are flagged immediately, but trust is rebuilt gradually** — exactly like real-world trust.

---

## 5. ML Models — Detailed Explanation

### 5.1 Variational Autoencoder (VAE) — "Digital Twin"

**Architecture**:
```
Input (14D) -> Linear(14->32) -> ReLU -> mu (32->16), logvar (32->16)
                                          | Reparameterization Trick
                                  z = mu + epsilon x sigma    (16D latent)
                                          |
                               Linear(16->32) -> ReLU -> Linear(32->14) -> Output
```

**Training**: One VAE is trained *per device* on its baseline behavior (50+ models total). Each model learns the probability distribution of that specific device's normal 14D feature vector.

**Scoring**: At runtime, the live feature vector is passed through the trained VAE. The **reconstruction error** (MSE between input and output) indicates anomaly level. High error = "this behavior doesn't match what I learned."

**Loss Function**: MSE (Reconstruction) + KL Divergence (Regularization)
```
L = MSE(x, x_hat) + KLD = MSE(x, x_hat) - 0.5 x Sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

### 5.2 Gaussian Mixture VAE (GMVAE) — "Hierarchical Twin"

An enhanced VAE with a **Gaussian Mixture Model** in the latent space. Instead of a single Gaussian, it learns **6 clusters** (one per device class: camera, sensor, thermostat, access_control, medical, industrial).

**6-Signal Comparison Engine**:
1. Reconstruction error (global model)
2. Reconstruction error (specialist model)
3. KL divergence from learned cluster centroid
4. Cosine drift from device's own historical latent centroid
5. Temporal variance of latent codes over last N evaluations
6. Cluster assignment confidence (softmax entropy)

Final deviation = weighted combination -> normalized to 0-1

### 5.3 Isolation Forest — "Statistical Outlier Detector"

- **Algorithm**: Builds random decision trees. Anomalous points are *easier to isolate* (fewer splits needed)
- **Training**: One model per device class (6 models total: `if_camera.joblib`, `if_sensor.joblib`, etc.)
- **Input**: Same 14D feature vector
- **Output**: `decision_function()` -> inverted and normalized to 0-1 anomaly score
- **Library**: scikit-learn

### 5.4 LSTM — "Temporal Sequence Predictor"

**Architecture**:
```
Input: Sequence of 12 feature snapshots (12 x 14D)
   -> LSTM(input=14, hidden=64, layers=2, dropout=0.2)
   -> Linear(64->14)
   -> Predicted next feature vector
```

**How it works**: The LSTM learns to *predict the next behavior snapshot* given the history. At runtime, it predicts what the device *should* do next. The **prediction error** (difference between predicted and actual) = anomaly score.

**Why it matters**: Catches **periodic patterns** like C2 beaconing (where a compromised device calls home to the attacker's server at regular intervals).

### 5.5 Graph Neural Network (GraphSAGE) — "Communication Topology Analyzer"

**Architecture**:
```
Node Features: 14D per device
Graph: Live communication graph (who talks to whom)
   -> SAGEConv(14->32) -> ReLU
   -> SAGEConv(32->32) -> ReLU
   -> Linear(32->2) -> Softmax [normal, anomalous]
```

**How it works**: Builds a **directed graph** where nodes = devices, edges = observed network communications. GraphSAGE aggregates features from *neighbors* to classify each node. If a sensor suddenly develops edges to 10 new devices (lateral movement), its neighborhood context changes and the GNN flags it.

**Runtime**: The graph is built incrementally as the trust engine evaluates devices. Edges expire after 5 minutes (sliding window).

---

## 6. The 14-Dimensional Feature Vector

Every 5-second window, the system extracts **14 features** from raw network flows:

| # | Feature | Description | Example Normal (Sensor) |
|---|---|---|---|
| 0 | `bytes_sent` | Total bytes transmitted | ~500 |
| 1 | `bytes_recv` | Total bytes received | ~200 |
| 2 | `packet_count` | Number of packets | ~20 |
| 3 | `bytes_per_packet` | Average payload size | ~25 |
| 4 | `upload_download_ratio` | Ratio of sent/received bytes | ~2.5 |
| 5 | `unique_dst_ips` | Distinct destination IPs | 2 (gateway + DNS) |
| 6 | `unique_dst_ports` | Distinct destination ports | 2 (1883 + 53) |
| 7 | `unique_src_ports` | Distinct source ports | 1 |
| 8 | `ext_int_ratio` | Fraction of traffic going external | ~0.01 |
| 9 | `active_hours_bitmap` | When the device is active | 24 (always on) |
| 10 | `inter_arrival_mean` | Average time between packets | ~30s |
| 11 | `inter_arrival_var` | Variance of inter-arrival times | ~2 |
| 12 | `burst_freq` | How often traffic bursts occur | ~0.5 |
| 13 | `protocol_entropy` | Shannon entropy of protocol distribution | ~0.65 |

---

## 7. DNA Fingerprinting — Device Identity Verification

The **DNA Fingerprint** is a **30-dimensional vector** constructed by concatenating:
- 14D raw feature vector (normalized)
- 16D latent vector from the VAE encoder (the "essence" of behavior)

**Uses**:
1. **Identity Verification**: Compare live DNA vs enrolled DNA using **Cosine Similarity**. If similarity drops below 0.85, the device may have been replaced or compromised with new firmware
2. **Unknown Device Classification**: Compare an unknown device's DNA against class-average DNAs to auto-classify it as camera/sensor/thermostat etc.

---

## 8. 5-Tier Autonomous Response System

When the trust score drops, the Response Engine automatically escalates through graduated tiers:

| Tier | Score Range | Action | Automatic? | Description |
|---|---|---|---|---|
| 1 | 80-100 | **Monitor** | Yes Auto | Normal operation. Recovery Manager releases restrictions |
| 2 | 60-79 | **Rate Limit** | Yes Auto | Bandwidth throttling via MQTT command (rate_delay_ms: 500) |
| 3 | 40-59 | **Sandbox** | Yes Auto | Redirect device to isolated VLAN |
| 4 | 20-39 | **Quarantine** | No HITL | Physical isolation. Queued for human approval (120s countdown, auto-executes if no response) |
| 5 | 0-19 | **Honeypot** | No HITL | Redirect to honeypot for forensic capture |

### HITL (Human-In-The-Loop)
For Tier 4 and 5 actions, the system:
1. Pushes a pending request to Redis with a 120-second countdown
2. Emits a Socket.IO event to the dashboard
3. The operator can **Approve**, **Deny**, or **Ignore**
4. If no response within 120s then auto-executes the action (safety fallback)

### Physical Actuation via ESP32
When the backend decides to quarantine a device, it sends an MQTT command:
```json
{
  "action": "quarantine",
  "trigger_score": 15.4,
  "relay_open": true,
  "rate_delay_ms": 0
}
```
The ESP32 receives this and **opens a physical relay** (GPIO 26 HIGH), literally disconnecting the device from the network.

---

## 9. Attack Scenarios — What DeviceDNA Detects

### Attack 1: Stealth Recon Scan (300s)
- **What**: Port scanning injected into camera/sensor traffic
- **Signature**: High `burst_freq`, high `packet_count` with low `bytes_per_packet`
- **Detection**: Isolation Forest + LSTM temporal anomaly
- **Real-world analog**: nmap scan from compromised IP camera

### Attack 2: Two-Stage Botnet C2 + DDoS (300s)
- **Stage 1 (120s)**: C2 beaconing on port 4444 — periodic connections to attacker's server
- **Stage 2 (180s)**: Volumetric UDP DDoS flood
- **Signature**: Low `inter_arrival_var` (periodic), high `protocol_entropy`, then massive `bytes_sent` spike
- **Detection**: LSTM catches beaconing pattern, VAE catches DDoS explosion
- **Real-world analog**: Mirai botnet

### Attack 3: Lateral Movement / Worm Spread (300s)
- **What**: Aggressively scanning internal peers on SSH/SMB/RDP ports
- **Signature**: Explosion in `unique_dst_ips` and `unique_dst_ports`, low `ext_int_ratio`
- **Detection**: GNN topology change, Isolation Forest outlier
- **Real-world analog**: WannaCry ransomware worm

### Attack 4: Massive Data Exfiltration (300s)
- **What**: Ransomware/spyware exfiltrating huge volumes over HTTPS
- **Signature**: `bytes_sent` > 5MB, high `ext_int_ratio`, anomalous `upload_download_ratio`
- **Detection**: CUSUM drift (gradual) + VAE (sudden), Isolation Forest outlier
- **Real-world analog**: APT data theft

---

## 10. Why These Databases?

### PostgreSQL (Relational)
- **Why**: ACID-compliant, structured data with relationships
- **Stores**: Device registry, alert records, response audit logs, policy rules, platform settings
- **Tables**: `devices`, `alerts`, `response_audit_logs`, `policy_rules`, `platform_settings`

### InfluxDB (Time-Series)
- **Why**: Optimized for timestamped data with automatic downsampling and retention policies
- **Stores**: Network telemetry data points, trust score history over time
- **Why not PostgreSQL for this?**: Millions of telemetry rows per day — InfluxDB handles time-range queries 10-100x faster

### Redis (In-Memory Key-Value)
- **Why**: Sub-millisecond reads/writes for real-time operations
- **Stores**: Live trust scores (`trust:{device_id}`), attack state (`attack_state:{device_id}`), HITL pending queue (`response:pending:{device_id}`), response action state
- **Why not PostgreSQL for this?**: Trust scores are recalculated every 5 seconds per device — needs in-memory speed

### Apache Kafka (Stream Processing)
- **Why**: High-throughput, fault-tolerant message queue for decoupled data ingestion
- **Topic**: `raw-flows` — all network telemetry (from simulator + physical devices) flows through this pipe
- **Why not direct HTTP?**: Kafka provides **buffering** (handles burst traffic), **replay** (can re-process data), and **decoupling** (producers and consumers run independently)

---

## 11. Protocols and Communication

| Protocol | Port | Used For | Where |
|---|---|---|---|
| **MQTT** | 1883 | ESP32 to Broker (telemetry + commands) | Mosquitto Broker |
| **HTTP/REST** | 8000 | Frontend to Backend API calls | FastAPI |
| **WebSocket (Socket.IO)** | 8000 | Real-time dashboard updates | FastAPI to Next.js |
| **Kafka Protocol** | 9092 | Stream processing (raw flows) | Kafka Broker |
| **PostgreSQL Wire** | 5432 | SQL queries | PostgreSQL |
| **InfluxDB Line Protocol** | 8086 | Time-series writes/queries | InfluxDB |
| **Redis RESP** | 6379 | Key-value operations | Redis |

### What is MQTT?
**Message Queuing Telemetry Transport** — a lightweight publish/subscribe messaging protocol designed for IoT.
- **Lightweight**: Minimum 2-byte header (vs HTTP's 700+ bytes)
- **Publish/Subscribe**: Devices publish to topics; subscribers listen
- **QoS Levels**: 0 (at most once), 1 (at least once), 2 (exactly once)
- **Retained Messages**: Broker stores the last message per topic
- **Topics in DeviceDNA**: `devicedna/flows/{device_id}`, `devicedna/{device_id}/command`, `devicedna/status/{device_id}`

### What is WebSocket / Socket.IO?
- **WebSocket**: Full-duplex communication channel over a single TCP connection (unlike HTTP which is request-response)
- **Socket.IO**: A library built on top of WebSocket that adds: automatic reconnection, room/namespace support, and fallback to HTTP long-polling
- **In DeviceDNA**: Used to push real-time trust score updates, alerts, and response actions from the backend to the dashboard without the frontend needing to poll

---

## 12. ESP32 Hardware Details

### ESP32 Dev Board (38-pin)
- **Processor**: Xtensa dual-core 32-bit LX6 at 240 MHz
- **RAM**: 520 KB SRAM
- **Flash**: 4 MB (external)
- **Wi-Fi**: 802.11 b/g/n (2.4 GHz)
- **Bluetooth**: v4.2 BR/EDR + BLE
- **GPIO Pins**: 34 programmable (out of 38 total pins including power)
- **ADC**: 18 channels (12-bit resolution)
- **DAC**: 2 channels (8-bit)
- **Operating Voltage**: 3.3V (powered via 5V micro-USB or VIN pin)
- **Communication**: UART, SPI, I2C, I2S, CAN, PWM
- **Total Pins on 38-pin board**: 38 pins = 2xGND, 1x3V3, 1xVIN, 34xGPIO

### Pin Mapping in DeviceDNA Firmware
| Pin | GPIO | Function |
|---|---|---|
| Relay Module | GPIO 26 | Controls physical relay (HIGH = open/disconnect, LOW = closed/connected) |
| Status LED | GPIO 32 | External indicator LED |
| RGB Red | GPIO 25 | Trust status: Red component |
| RGB Green | GPIO 33 | Trust status: Green component |
| RGB Blue | GPIO 27 | Trust status: Blue component |

### RGB LED Trust Indicator
| Trust Score | LED Color | Meaning |
|---|---|---|
| >= 70 | Solid Green | Trusted |
| 40-69 | Yellow (Red+Green) | Guarded/Suspicious |
| < 40 | Solid Red | Critical / Under Attack |

### Libraries Used on ESP32
| Library | Purpose |
|---|---|
| `WiFi.h` | Wi-Fi connectivity (built into ESP32 SDK) |
| `PubSubClient.h` | MQTT client for publishing telemetry and subscribing to commands |
| `ArduinoJson.h` | JSON serialization/deserialization of telemetry payloads |
| `time.h` | NTP time synchronization for accurate timestamps |

---

## 13. Use Cases

### 1. Smart Hospital Network
Protect medical IoT devices (patient monitors, infusion pumps, imaging systems) from being compromised. Automatically quarantine a rogue infusion pump before it can exfiltrate patient data.

### 2. Industrial SCADA / ICS
Monitor PLCs, sensors, and actuators in manufacturing plants. Detect a compromised temperature sensor that has been reprogrammed to report false readings.

### 3. Smart Building / Campus
Secure hundreds of IP cameras, access control panels, HVAC systems. Detect lateral movement from a compromised camera to the door lock system.

### 4. Critical Infrastructure
Water treatment plants, power grids, telecom towers — where IoT compromise can have physical safety consequences. DeviceDNA's physical relay isolation provides a last-resort hardware kill switch.

### 5. Enterprise IoT Fleet Management
Organizations deploying thousands of IoT devices across multiple sites. Centralized trust monitoring with per-device behavioral baselines.

---

## 14. Methodology — End-to-End Workflow

```
Step 1: ENROLLMENT
   |-> Register device in PostgreSQL (MAC, IP, class, VLAN, firmware)
   |-> Start baseline collection period (~24 hours of normal traffic)

Step 2: BASELINE LEARNING
   |-> Telemetry flows through Kafka -> Feature Extraction -> 14D vectors
   |-> Train per-device VAE, per-class Isolation Forest, shared LSTM, shared GNN
   |-> Store trained models (.pt, .joblib) + normalization params (.json)

Step 3: REAL-TIME MONITORING
   |-> Every 5 seconds:
       |-> Consume raw flows from Kafka topic "raw-flows"
       |-> Extract 14D feature vector
       |-> Run 5-pillar scoring (VAE, IF, LSTM, GNN, CUSUM, Policy, Peer)
       |-> Compute weighted trust score (0-100)
       |-> Apply EMA smoothing + trust decay
       |-> Store in Redis (real-time) + InfluxDB (historical)
       |-> Emit via Socket.IO to dashboard

Step 4: THREAT DETECTION
   |-> Trust score drops below threshold
   |-> Generate Threat Intelligence Brief (TIB) with SHAP explainability
   |-> Store alert in PostgreSQL

Step 5: AUTONOMOUS RESPONSE
   |-> Response Engine maps score to Tier (1-5)
   |-> Tiers 2-3: Auto-execute (rate limit, sandbox)
   |-> Tiers 4-5: Queue for HITL approval (120s countdown)
   |-> Send MQTT command to ESP32 relay (quarantine/recover)
   |-> Log to PostgreSQL audit trail

Step 6: RECOVERY
   |-> When attack ends and trust score rises back above 80
   |-> Recovery Manager releases restrictions
   |-> ESP32 relay closes (device reconnected)
   |-> RGB LED turns green
```

---

## 15. Technology Stack Summary

| Layer | Technology | Version |
|---|---|---|
| **Backend** | Python, FastAPI, Uvicorn | Python 3.11 |
| **ML Framework** | PyTorch, scikit-learn, NetworkX | PyTorch 2.x |
| **Frontend** | Next.js, React, TailwindCSS, Framer Motion | Next.js 14 |
| **Real-time** | Socket.IO (python-socketio) | -- |
| **Databases** | PostgreSQL, InfluxDB, Redis | PG 16, Influx 2.7, Redis 7 |
| **Streaming** | Apache Kafka + Zookeeper | Confluent 7.5 |
| **IoT Protocol** | MQTT (Mosquitto broker) | Eclipse 2.0 |
| **Hardware** | ESP32 Dev Board (38-pin) | -- |
| **Containerization** | Docker, Docker Compose | -- |
| **Firmware** | Arduino C++ (PubSubClient, ArduinoJson) | -- |

---

## 16. Viva Q&A — Probable Questions and Answers

### General IoT Questions

**Q: What is IoT?**
A: Internet of Things — a network of physical devices embedded with sensors, software, and connectivity that enables them to collect, exchange, and act on data. Examples: smart cameras, thermostats, wearables, industrial sensors.

**Q: What are the common IoT communication protocols?**
A: MQTT, CoAP, HTTP/HTTPS, AMQP, Zigbee, Z-Wave, BLE, LoRa, and WebSocket. DeviceDNA primarily uses **MQTT** (for devices) and **HTTP/WebSocket** (for dashboard).

**Q: Why is MQTT preferred over HTTP in IoT?**
A: MQTT has a 2-byte minimum header (vs HTTP's 700+ bytes), supports publish/subscribe pattern (event-driven vs polling), has QoS levels, persistent sessions, and last-will messages. Ideal for constrained devices with limited bandwidth.

**Q: How many pins does ESP32 have?**
A: The ESP32 38-pin development board has **38 total pins**: 34 GPIO pins + 2 GND + 1 VIN + 1 3.3V. Some GPIO pins are input-only (GPIO 34-39). Our project uses GPIO 25, 26, 27, 32, and 33.

**Q: What is the difference between ESP32 and ESP8266?**
A: ESP32 is the successor — it has dual-core CPU (vs single), Bluetooth (ESP8266 has none), more GPIO pins (34 vs 17), hardware encryption, and higher clock speed (240 MHz vs 160 MHz).

**Q: What is a relay module?**
A: An electrically operated switch. When GPIO 26 goes HIGH, the relay opens (circuit breaks), physically disconnecting the device from the network. This is our "hardware kill switch" for quarantine.

### Architecture Questions

**Q: What is WebSocket?**
A: A full-duplex communication protocol that provides persistent connection between client and server. Unlike HTTP (request-response), WebSocket allows the server to push data to the client in real-time. We use Socket.IO (built on WebSocket) to push live trust scores and alerts to the dashboard.

**Q: Why use Kafka instead of direct HTTP?**
A: Kafka provides: (1) **Decoupling** — producers and consumers are independent, (2) **Buffering** — handles traffic bursts without data loss, (3) **Replay** — can re-process historical data, (4) **Scalability** — handles millions of messages/second.

**Q: Why three databases?**
A: Each database excels at a different data pattern: PostgreSQL for structured relational data (ACID compliance), InfluxDB for time-series data (10-100x faster for time-range queries), Redis for sub-millisecond real-time operations (trust scores updated every 5 seconds per device).

**Q: What is Docker?**
A: A containerization platform that packages applications with their dependencies into isolated containers. Our `docker-compose.yml` defines all 9 services and their relationships, allowing single-command deployment with `docker-compose up`.

**Q: What is CORS?**
A: Cross-Origin Resource Sharing — a browser security mechanism that restricts web pages from making requests to a different domain. Our backend has CORS middleware configured to allow the frontend (port 3000) to call the backend API (port 8000).

### ML Questions

**Q: What is a VAE?**
A: Variational Autoencoder — a generative neural network that learns to encode data into a compressed latent space (encoder) and reconstruct it back (decoder). The "variational" part means it learns a *probability distribution* (mu, sigma squared) rather than fixed points, enabling better generalization. We use reconstruction error as an anomaly score.

**Q: What is the reparameterization trick?**
A: z = mu + epsilon x sigma, where epsilon is sampled from N(0,1). This makes the sampling step differentiable, allowing backpropagation through the stochastic sampling layer during training.

**Q: What is Isolation Forest?**
A: An unsupervised anomaly detection algorithm that isolates observations by randomly selecting features and split values. Anomalies require fewer splits to isolate (shorter path length) than normal points. Time complexity: O(n log n).

**Q: What is LSTM?**
A: Long Short-Term Memory — a type of Recurrent Neural Network (RNN) with gates (forget, input, output) that can learn long-term dependencies in sequential data. We use it to predict the next feature vector from a history of 12 snapshots. Prediction error = anomaly.

**Q: What is GNN / GraphSAGE?**
A: Graph Neural Network that operates on graph-structured data. GraphSAGE (SAmple and agGrEgate) learns node representations by *sampling and aggregating features from neighbors*. We build a live communication graph (who talks to whom) and classify nodes as normal/anomalous based on their neighborhood context.

**Q: What is CUSUM?**
A: Cumulative Sum — a sequential analysis technique for detecting persistent shifts in the mean of a process. It accumulates z-scored deviations from baseline. If the cumulative sum exceeds threshold H, it signals an alarm. Key advantage: detects *slow, gradual* changes that point-anomaly detectors miss.

**Q: What is SHAP?**
A: SHapley Additive exPlanations — a game-theoretic approach to explain ML model predictions. It assigns each feature an importance value (Shapley value) that quantifies its contribution to the prediction. We use it to generate human-readable explanations in our Threat Intelligence Briefs.

**Q: How do you prevent false positives?**
A: Multiple mechanisms: (1) EMA smoothing with asymmetric alpha, (2) Trust decay multiplier for repeated offenders, (3) 5-pillar consensus (no single model can tank the score alone), (4) Peer comparison against same-class devices, (5) HITL approval for critical actions (Tier 4-5).

### Security Questions

**Q: What types of attacks can DeviceDNA detect?**
A: Recon/port scanning, Botnet C2 beaconing, DDoS floods, lateral movement/worm spread, data exfiltration (both sudden and slow), firmware tampering (via DNA fingerprint comparison), and unknown/zero-day attacks (because we detect deviation from normal, not known signatures).

**Q: How is this different from a traditional IDS?**
A: Traditional IDS (e.g., Snort) uses **signature matching** — it compares traffic against a database of known attack patterns. DeviceDNA uses **behavioral anomaly detection** — it learns normal per-device behavior and detects *any* deviation, including zero-day attacks that have no known signature.

**Q: What is Zero Trust?**
A: A security model where no device or user is trusted by default, even if they are inside the network perimeter. DeviceDNA implements Zero Trust by: (1) continuously verifying device behavior, (2) enforcing least-privilege communication policies, (3) dynamically revoking trust when anomalies are detected.

**Q: What is the CIA triad?**
A: **Confidentiality** (data is private), **Integrity** (data is accurate and unmodified), **Availability** (systems are accessible when needed). DeviceDNA protects all three: detects exfiltration (C), firmware tampering (I), and DDoS (A).

### Project-Specific Questions

**Q: How many devices does the system support?**
A: Currently simulates **50 virtual devices** across 6 classes + **physical ESP32 nodes**. Architecture is horizontally scalable — Kafka handles millions of flows, and each ML model is lightweight (~14KB per VAE).

**Q: What is the latency from attack to detection?**
A: ~5-10 seconds. The telemetry consumer processes flows every 5 seconds, runs all 5 ML pillars, computes trust score, and emits via Socket.IO in a single cycle.

**Q: What happens if the backend goes down?**
A: (1) The simulator buffers flows in Kafka (retention: 1 hour), (2) ESP32 devices continue operating with their last-known state, (3) When backend restarts, Kafka consumer picks up from the latest offset and processing resumes.

**Q: What is the DNA fingerprint?**
A: A 30-dimensional vector (14D raw features + 16D VAE latent encoding) that uniquely represents a device's behavioral identity. Used for: identity verification (cosine similarity), unknown device classification, and firmware tampering detection.

---

> **Tip for the viva**: If asked about something you are unsure of, relate it back to the trust score formula — it is the heart of the entire system. Everything else (ML models, databases, ESP32 actuation) feeds into or acts upon that score.
