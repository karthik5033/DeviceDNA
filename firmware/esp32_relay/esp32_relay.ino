/**
 * DeviceDNA ESP32 Relay Node Firmware
 * Actuates physical isolation commands received via MQTT.
 * 
 * Target Hardware: ESP32 Dev Board (38-pin)
 * Relay Module: 5V Relay Connected to GPIO 26
 * RGB Status Indicator: GPIO 25 (Red), GPIO 33 (Green), GPIO 27 (Blue)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ── Wi-Fi & MQTT Configuration ──────────────────────────────────────────────
const char* ssid          = "DeviceDNA_Lab";
const char* password      = "devicedna2025";
const char* mqtt_broker   = "192.168.43.1"; // Host laptop IP address
const int   mqtt_port     = 1883;

// Device Identity
const char* device_id     = "ir_sensor";
const char* device_type   = "gateway";

// MQTT Topics
const char* topic_telemetry = "devicedna/telemetry";
const char* topic_command   = "devicedna/ir_sensor/command";
const char* topic_status    = "devicedna/status/ir_sensor";

// ── Pin Mappings ────────────────────────────────────────────────────────────
#define RELAY_PIN       26  // Pin triggering the relay module
#define STATUS_LED_PIN  32  // External indicator LED
#define LED_RED_PIN     25  // RGB Red
#define LED_GREEN_PIN   33  // RGB Green
#define LED_BLUE_PIN    27  // RGB Blue

// Global State
WiFiClient espClient;
PubSubClient mqttClient(espClient);
unsigned long lastTelemetry = 0;
bool g_quarantined = false;
int g_rate_delay_ms = 0;

void setTrustLED(float score) {
  if (score >= 70.0) {
    // Trusted: Solid Green
    digitalWrite(LED_RED_PIN,   LOW);
    digitalWrite(LED_GREEN_PIN, HIGH);
    digitalWrite(LED_BLUE_PIN,  LOW);
  } else if (score >= 40.0) {
    // Guarded/Suspicious: Yellow (Red + Green)
    digitalWrite(LED_RED_PIN,   HIGH);
    digitalWrite(LED_GREEN_PIN, HIGH);
    digitalWrite(LED_BLUE_PIN,  LOW);
  } else {
    // Critical / Under Attack: Solid Red
    digitalWrite(LED_RED_PIN,   HIGH);
    digitalWrite(LED_GREEN_PIN, LOW);
    digitalWrite(LED_BLUE_PIN,  LOW);
  }
}

// ── MQTT Subscription Callback ──────────────────────────────────────────────
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.println(topic);

  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, payload, length);
  
  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.f_str());
    return;
  }

  const char* action = doc["action"];
  float score = doc["trigger_score"] | 100.0;
  g_rate_delay_ms = doc["rate_delay_ms"] | 0;

  Serial.print("Action: "); Serial.println(action);
  Serial.print("Trigger Score: "); Serial.println(score);

  setTrustLED(score);

  if (strcmp(action, "quarantine") == 0) {
    g_quarantined = true;
    digitalWrite(RELAY_PIN, HIGH);   // Open relay: physical network isolation
    digitalWrite(STATUS_LED_PIN, LOW); // Status indicator OFF
    Serial.println("[RELAY] GPIO 26: HIGH (Relay open, circuit broken) - Device Quarantined!");
  }
  else if (strcmp(action, "rate_limit") == 0) {
    g_quarantined = false;
    digitalWrite(RELAY_PIN, LOW);    // Keep connected
    digitalWrite(STATUS_LED_PIN, HIGH);
    Serial.print("[THROTTLE] Active: introducing latency delay of ");
    Serial.print(g_rate_delay_ms);
    Serial.println("ms.");
  }
  else if (strcmp(action, "recover") == 0 || strcmp(action, "release") == 0) {
    g_quarantined = false;
    g_rate_delay_ms = 0;
    digitalWrite(RELAY_PIN, LOW);    // Close relay: reconnect circuit
    digitalWrite(STATUS_LED_PIN, HIGH); // Status indicator ON
    Serial.println("[RELAY] GPIO 26: LOW (Relay closed, circuit connected) - Device recovered.");
  }
}

void connectWiFi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to Wi-Fi SSID: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("Wi-Fi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  mqttClient.setServer(mqtt_broker, mqtt_port);
  mqttClient.setCallback(mqttCallback);

  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection to broker: ");
    Serial.print(mqtt_broker);
    Serial.print("...");

    if (mqttClient.connect(device_id)) {
      Serial.println("connected!");
      
      // Subscribe to command channel
      mqttClient.subscribe(topic_command);
      
      // Publish initial registration message
      StaticJsonDocument<128> reg;
      reg["device_id"] = device_id;
      reg["device_type"] = device_type;
      reg["status"] = "online";
      char buf[128];
      serializeJson(reg, buf);
      mqttClient.publish(topic_status, buf);
      
      Serial.println("Subscribed and registered device status.");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" trying again in 5 seconds...");
      delay(5000);
    }
  }
}

// ── Setup & Loop ────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_BLUE_PIN, OUTPUT);

  // Initialize output pin states
  digitalWrite(RELAY_PIN, LOW);      // Close relay on boot (connected)
  digitalWrite(STATUS_LED_PIN, HIGH); // Status LED ON
  setTrustLED(100.0);                // RGB Green

  connectWiFi();
  connectMQTT();
}

void loop() {
  if (!mqttClient.connected()) {
    connectMQTT();
  }
  mqttClient.loop();

  unsigned long now = millis();
  // Send telemetry every 5 seconds if not quarantined
  if (now - lastTelemetry >= 5000) {
    lastTelemetry = now;

    if (!g_quarantined) {
      StaticJsonDocument<256> telemetry;
      telemetry["device_id"] = device_id;
      telemetry["device_type"] = device_type;
      telemetry["timestamp"] = now / 1000;
      
      // Simulate normal gateway telemetry
      telemetry["total_flows"] = random(20, 35);
      telemetry["total_bytes"] = random(1000, 2500);
      telemetry["avg_packet_size"] = random(150, 300);
      telemetry["external_ratio"] = 0.08;
      telemetry["https_ratio"] = 0.50;
      telemetry["tcp_ratio"] = 0.90;
      telemetry["unique_dst_ips"] = random(2, 6);
      telemetry["unique_dst_ports"] = random(2, 4);

      char buf[256];
      serializeJson(telemetry, buf);
      mqttClient.publish(topic_telemetry, buf);
      Serial.println("[TX] Telemetry report published to broker.");
    } else {
      Serial.println("[QUARANTINE] Telemetry transmission blocked.");
    }

    // Handle rate-limiting simulated delay
    if (g_rate_delay_ms > 0) {
      delay(g_rate_delay_ms);
    }
  }
}
