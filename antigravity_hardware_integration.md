# DeviceDNA — Antigravity Hardware Integration Guide
# 5 Physical ESP32 Nodes → DeviceDNA Trust Engine

> This document is the AI-agent reference for integrating 5 physical ESP32 sensor nodes into the existing DeviceDNA software pipeline. It replaces 5 virtual devices from the 46-device simulator with real hardware publishing identical telemetry over MQTT. The sensor type on each ESP32 is irrelevant to this layer — what matters is the MQTT topic schema, 14D feature vector output, device_id mapping, and trust engine registration.

---

## 1. Integration Philosophy

The DeviceDNA pipeline is sensor-agnostic at the network layer. Every device — physical or virtual — is reduced to the same **14-Dimensional Feature Vector** before it touches any ML model. This means:

- The ESP32 firmware only needs to publish periodic MQTT payloads with the correct JSON schema.
- The FastAPI consumer doesn't care whether a flow came from a Python asyncio coroutine or a physical ESP32.
- The Trust Engine scores all 50 devices (5 physical + 45 virtual remaining) identically.

The only integration work is:
1. Assigning correct `device_id` and `device_class` to each physical ESP32.
2. Retiring the corresponding 5 virtual devices from `simulator.py`.
3. Ensuring MQTT topic and payload schema match exactly.
4. Handling the per-device VAE model — physical devices need their own trained `.pt` file.

---

## 2. Physical Device Roster — 5 ESP32 Nodes

| Node | Device ID | Device Class | Sensor | MQTT Topic | Replaces Virtual |
|------|-----------|--------------|--------|------------|-----------------|
| ESP32 #1 | `HW-001` | `sensor` | DHT11 (Temp/Humidity) | `devicedna/flows/HW-001` | `SIM-001` |
| ESP32 #2 | `HW-002` | `sensor` | DHT11 (Temp/Humidity) | `devicedna/flows/HW-002` | `SIM-002` |
| ESP32 #3 | `HW-003` | `access_control` | IR Sensor | `devicedna/flows/HW-003` | `SIM-003` |
| ESP32 #4 | `HW-004` | `industrial` | Current Sensor | `devicedna/flows/HW-004` | `SIM-004` |
| ESP32 #5 | `HW-005` | `sensor` | LDR (Light) | `devicedna/flows/HW-005` | `SIM-005` |

> **Device class assignment rationale:**
> - `sensor` — periodic low-frequency telemetry (temp, humidity, light). Low byte volume, predictable intervals.
> - `access_control` — event-triggered bursts (IR detects motion/presence). Irregular timing is normal.
> - `industrial` — continuous monitoring, higher byte rates (current draw logging).

---

## 3. MQTT Payload Schema — What Each ESP32 Must Publish

Every ESP32 publishes to its topic at a fixed interval. The payload must exactly match the raw flow schema consumed by `backend/app/services/telemetry.py`.

### 3.1 Required JSON Payload

```json
{
  "flow_id": "<uuid-generated-on-device>",
  "src_ip": "<esp32-ip-on-hotspot-lan>",
  "dst_ip": "<server-laptop-ip>",
  "src_port": 1883,
  "dst_port": 8000,
  "protocol": "MQTT",
  "bytes": <sensor_reading_mapped_to_bytes>,
  "packets": <publish_count_since_last_interval>,
  "duration": <interval_in_seconds>,
  "timestamp": "<ISO8601-UTC>",
  "device_id": "HW-00X",
  "device_class": "<sensor|access_control|industrial>"
}
```

### 3.2 Field Mapping — Physical Sensor → Network Telemetry Fields

Since ESP32s don't generate real network flows, sensor readings are mapped to telemetry fields that the feature extractor understands:

| Telemetry Field | Physical Mapping |
|-----------------|-----------------|
| `bytes` | Scale sensor reading to a realistic byte range for the device class. DHT11: `humidity * 100 + temp * 10`. IR: `1500` on trigger, `50` idle. Current: `amps * 1000`. LDR: `lux_value * 2`. |
| `packets` | Fixed: publish count since last report (usually 1–5). |
| `duration` | Reporting interval in seconds (e.g., `5.0` for 5-second intervals). |
| `src_ip` | ESP32's actual IP assigned by the mobile hotspot. |
| `dst_ip` | Server laptop's IP on the hotspot LAN (e.g., `192.168.43.XXX`). |
| `protocol` | Hardcode `"MQTT"`. |
| `flow_id` | Generate on-device using `esp_random()` cast to hex string, or a counter string. |
| `timestamp` | NTP-synced UTC time. Use `configTime()` with pool.ntp.org in firmware. |

---

## 4. Firmware Requirements (Arduino C++)

The firmware for all 5 boards shares the same structure. The only differences per board are `DEVICE_ID`, `DEVICE_CLASS`, and the sensor read logic that maps to the `bytes` field.

### 4.1 Required Libraries (Arduino IDE)

```
WiFi.h              — built-in ESP32
PubSubClient        — MQTT client (install via Library Manager)
ArduinoJson         — JSON serialization (install via Library Manager)
time.h              — NTP sync (built-in)
```

### 4.2 Shared Config Block (config.h)

```cpp
// config.h — shared across all 5 boards, change per board before flashing

#define WIFI_SSID        "YourHotspotSSID"
#define WIFI_PASSWORD    "YourHotspotPassword"
#define MQTT_BROKER      "192.168.43.XXX"   // Server laptop IP — run ipconfig to get this
#define MQTT_PORT        1883
#define DEVICE_ID        "HW-001"           // Change per board: HW-001 to HW-005
#define DEVICE_CLASS     "sensor"           // Change per board
#define REPORT_INTERVAL  5000               // Milliseconds between publishes
#define MQTT_TOPIC       "devicedna/flows/HW-001"  // Change per board
```

### 4.3 Core Firmware Loop Structure

```cpp
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>
#include "config.h"

WiFiClient espClient;
PubSubClient mqttClient(espClient);
int publishCount = 0;

void setup() {
  Serial.begin(115200);
  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  syncNTP();
  initSensor();   // Board-specific sensor init
}

void loop() {
  if (!mqttClient.connected()) reconnectMQTT();
  mqttClient.loop();

  static unsigned long lastReport = 0;
  if (millis() - lastReport >= REPORT_INTERVAL) {
    lastReport = millis();
    publishFlow();
  }
}

void publishFlow() {
  float sensorReading = readSensor();   // Board-specific
  long bytesVal = mapSensorToBytes(sensorReading);  // Board-specific
  publishCount++;

  // Get timestamp
  time_t now;
  struct tm timeinfo;
  time(&now);
  gmtime_r(&now, &timeinfo);
  char timestamp[30];
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);

  // Get src IP
  String srcIP = WiFi.localIP().toString();

  // Build flow_id (simple counter)
  char flowId[20];
  snprintf(flowId, sizeof(flowId), "%s-%06d", DEVICE_ID, publishCount);

  // Serialize JSON
  StaticJsonDocument<256> doc;
  doc["flow_id"]      = flowId;
  doc["src_ip"]       = srcIP;
  doc["dst_ip"]       = MQTT_BROKER;
  doc["src_port"]     = 1883;
  doc["dst_port"]     = 8000;
  doc["protocol"]     = "MQTT";
  doc["bytes"]        = bytesVal;
  doc["packets"]      = publishCount % 5 + 1;
  doc["duration"]     = REPORT_INTERVAL / 1000.0;
  doc["timestamp"]    = timestamp;
  doc["device_id"]    = DEVICE_ID;
  doc["device_class"] = DEVICE_CLASS;

  char payload[256];
  serializeJson(doc, payload);
  mqttClient.publish(MQTT_TOPIC, payload);
  Serial.println(payload);
}
```

### 4.4 Per-Board Sensor Functions

**HW-001 & HW-002 — DHT11 (sensor class)**
```cpp
#include <DHT.h>
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

void initSensor() { dht.begin(); }

float readSensor() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (isnan(h) || isnan(t)) return 500.0;
  return h;
}

long mapSensorToBytes(float reading) {
  float t = dht.readTemperature();
  return (long)(reading * 100 + t * 10);  // e.g., 65% humidity + 25°C = 6750 bytes
}
```

**HW-003 — IR Sensor (access_control class)**
```cpp
#define IR_PIN 14

void initSensor() {
  pinMode(IR_PIN, INPUT);
}

float readSensor() {
  return digitalRead(IR_PIN);  // 1 = motion detected, 0 = idle
}

long mapSensorToBytes(float reading) {
  return reading > 0 ? 1500 : 50;  // Burst on trigger, minimal idle
}
```

**HW-004 — Current Sensor (industrial class)**
```cpp
#define CURRENT_PIN 34  // ADC pin

void initSensor() {
  analogReadResolution(12);
}

float readSensor() {
  int raw = analogRead(CURRENT_PIN);
  return (raw / 4095.0) * 30.0;  // Scale to 0–30A range
}

long mapSensorToBytes(float reading) {
  return (long)(reading * 1000);  // e.g., 5.5A = 5500 bytes
}
```

**HW-005 — LDR (sensor class)**
```cpp
#define LDR_PIN 35  // ADC pin

void initSensor() {
  analogReadResolution(12);
}

float readSensor() {
  int raw = analogRead(LDR_PIN);
  return (raw / 4095.0) * 1000.0;  // Scale to lux-like 0–1000
}

long mapSensorToBytes(float reading) {
  return (long)(reading * 2);
}
```

---

## 5. Simulator Changes — Retiring 5 Virtual Devices

In `simulator.py` (or the equivalent virtual device module), comment out or remove the 5 virtual devices being replaced.

### 5.1 Devices to Retire

```python
# In simulator.py — remove or comment out these device definitions:
# { "device_id": "SIM-001", "device_class": "sensor", ... }
# { "device_id": "SIM-002", "device_class": "sensor", ... }
# { "device_id": "SIM-003", "device_class": "access_control", ... }
# { "device_id": "SIM-004", "device_class": "industrial", ... }
# { "device_id": "SIM-005", "device_class": "sensor", ... }

# Fleet now: 41 virtual + 5 physical = 46 total devices
```

### 5.2 Device Count After Integration

| Source | Count | Notes |
|--------|-------|-------|
| Physical ESP32 nodes | 5 | HW-001 to HW-005 |
| Virtual Python devices | 41 | Remaining after retiring 5 |
| **Total** | **46** | Same as before — trust engine unchanged |

---

## 6. Backend Changes — Registering Physical Devices

### 6.1 MQTT Consumer (`backend/app/services/telemetry.py`)

No code changes needed. The FastAPI MQTT consumer already subscribes to `devicedna/flows/#` (wildcard). Physical ESP32s publishing to `devicedna/flows/HW-00X` are automatically picked up.

Verify the subscription wildcard is in place:
```python
# In telemetry.py MQTT setup — confirm this exists:
await client.subscribe("devicedna/flows/#")
```

### 6.2 Redis Device Registration

Physical devices are auto-registered in Redis the first time a flow is processed. The key `trust:HW-001` etc. is created on first score computation. No manual seeding needed.

### 6.3 Per-Device VAE Models

Each device needs a trained `.pt` VAE model file in `models_trained/`. Physical devices start with no history, so:

**Option A (Recommended for demo):** Bootstrap with the corresponding virtual device's model.
```bash
# Copy the retired virtual device's model as the physical device's starting model
cp models_trained/SIM-001.pt models_trained/HW-001.pt
cp models_trained/SIM-002.pt models_trained/HW-002.pt
cp models_trained/SIM-003.pt models_trained/HW-003.pt
cp models_trained/SIM-004.pt models_trained/HW-004.pt
cp models_trained/SIM-005.pt models_trained/HW-005.pt
```

**Option B (If time allows):** Run physical devices for ~30 minutes, then retrain:
```bash
python train_vae.py --device_ids HW-001 HW-002 HW-003 HW-004 HW-005
```

### 6.4 Policy Conformance Rules (`backend/app/services/trust_engine.py`)

The existing policy rules are defined per `device_class`, not `device_id`, so physical devices inherit the correct rules automatically. Confirm the classes map correctly:

| Physical Node | Device Class | Policy Rules Applied |
|--------------|--------------|---------------------|
| HW-001, HW-002, HW-005 | `sensor` | Low byte volume, predictable intervals |
| HW-003 | `access_control` | Event-burst tolerance, low sustained traffic |
| HW-004 | `industrial` | Higher byte rate allowed, continuous stream |

---

## 7. Network Architecture — Physical Devices on Hotspot LAN

All 5 ESP32 boards join the same mobile hotspot network as the server and attacker laptops.

```
Mobile Hotspot (192.168.43.1)
│
├── Server Laptop     192.168.43.XXX   (Mosquitto :1883, FastAPI :8000/:8001)
├── Attacker Laptop   192.168.43.YYY
├── ESP32 HW-001      192.168.43.A     (DHT11)
├── ESP32 HW-002      192.168.43.B     (DHT11)
├── ESP32 HW-003      192.168.43.C     (IR)
├── ESP32 HW-004      192.168.43.D     (Current)
└── ESP32 HW-005      192.168.43.E     (LDR)
```

IPs are DHCP-assigned by the hotspot. Before the demo:
1. Run `ipconfig` on server laptop — note the IP.
2. Check Mosquitto logs to see which IPs the ESP32s connected from.
3. Update `config.h` `MQTT_BROKER` with the exact server IP and reflash if changed.

**Tip:** Reserve IPs if your hotspot supports it, or note them the day before the demo.

---

## 8. Trust Score Behavior — Physical vs Virtual

Physical ESP32 data will produce slightly different trust scores than the virtual simulator because:

- Real sensor readings have natural noise (no smooth sine-wave patterns).
- Network timing is real MQTT latency, not a controlled asyncio loop.
- Temperature/IR/current readings produce organic `bytes` values vs simulator ranges.

This is **a feature, not a bug** — the physical nodes will look authentically different on the dashboard, making the demo more convincing to judges.

Expected behavior on a clean physical node:
- Trust score settles between **78–92** (slight anomaly from bootstrap model mismatch).
- Score stabilizes after 10–15 minutes as the decay/recovery engine adapts.
- No alerts should fire on idle physical nodes.

---

## 9. Dashboard Impact

The React dashboard requires no changes. Physical nodes appear identically to virtual ones:
- They show up in the Force-Graph as nodes with their `device_id` (HW-001 etc.).
- Trust score line charts update on `trust_update` WebSocket events.
- The `device_class` drives the node color/icon in the topology graph.

To visually distinguish physical from virtual in the demo, consider adding a `"source": "physical"` field to the MQTT payload and filtering in the dashboard — but this is optional and not required for the trust engine.

---

## 10. Pre-Demo Checklist — Physical Hardware

```
[ ] All 5 ESP32 boards flashed with correct DEVICE_ID and DEVICE_CLASS
[ ] MQTT_BROKER IP in config.h matches server laptop's hotspot IP
[ ] All boards connected to hotspot WiFi (check Serial Monitor)
[ ] Mosquitto logs show 5 new client connections (HW-001 to HW-005)
[ ] MQTT Explorer or mosquitto_sub shows payloads arriving on devicedna/flows/HW-00X
[ ] Redis keys trust:HW-001 through trust:HW-005 exist after ~30 seconds
[ ] Dashboard shows 46 nodes total (41 virtual + 5 physical)
[ ] Trust scores for HW-00X visible in dashboard panel
[ ] VAE model files exist: models_trained/HW-001.pt through HW-005.pt
[ ] Retiring of SIM-001 to SIM-005 confirmed in simulator.py
```

---

## 11. Troubleshooting — Physical Node Issues

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| ESP32 not connecting to WiFi | Wrong SSID/password in config.h | Reflash with corrected credentials. SSID is case-sensitive. |
| ESP32 connects but MQTT fails | Server IP changed after hotspot reconnect | Run `ipconfig` on server, update `MQTT_BROKER`, reflash. |
| `trust:HW-00X` never appears in Redis | Payload schema mismatch | Check Serial Monitor output vs required JSON schema in Section 3. |
| Trust score stuck at 0 | VAE model file missing | Copy from retired SIM model (Section 6.3 Option A). |
| Physical node trust score always 100 | Policy rules too loose for device class | Verify `device_class` in payload matches the intended class. |
| Sensor returns NaN (DHT11) | Missing 10k pull-up resistor | Add 10k between data pin and 3.3V. Try a different GPIO. |
| Dashboard shows 41 nodes, not 46 | SIM devices retired but HW not publishing | Confirm ESP32s are on hotspot and MQTT payloads arriving. |

---

*DeviceDNA — Antigravity Hardware Integration Guide | 5 Physical ESP32 Nodes | Eclipse Hackathon 2025*
