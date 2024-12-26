#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>

// WiFi Credentials
const char* ssid = "Wokwi-GUEST";
const char* password = "";

// ThingsBoard MQTT Credentials
const char* mqtt_server = "demo.thingsboard.io";
const char* token = "hjAUXQxLQ6vDL6J8EwJA";

// MQTT Client
WiFiClient espClient;
PubSubClient client(espClient);

// Pin Definitions
#define DHTPIN 15      // Pin connected to DHT22
#define DHTTYPE DHT22  // DHT22 sensor
#define LDRPIN 17      // Pin connected to LDR
#define METHANE_PIN 36 // Potentiometer for methane gas
#define TOXIN_PIN 39   // Potentiometer for toxin gas
#define RELAY1 21      // Relay 1 for temperature
#define RELAY2 20      // Relay 2 for humidity
#define RELAY_METHANE 19 // Relay for methane gas
#define RELAY_TOXIN 18   // Relay for toxin gas
#define BUZZER 10      // Buzzer for LDR

// Sensor and Actuator Initialization
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(LDRPIN, INPUT);
  pinMode(METHANE_PIN, INPUT);
  pinMode(TOXIN_PIN, INPUT);
  pinMode(RELAY1, OUTPUT);
  pinMode(RELAY2, OUTPUT);
  pinMode(RELAY_METHANE, OUTPUT);
  pinMode(RELAY_TOXIN, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  digitalWrite(RELAY1, LOW);
  digitalWrite(RELAY2, LOW);
  digitalWrite(RELAY_METHANE, LOW);
  digitalWrite(RELAY_TOXIN, LOW);
  digitalWrite(BUZZER, LOW);

  setupWiFi();
  client.setServer(mqtt_server, 1883);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Simulate Sensor Values
  float temperature = 28 + random(-5, 6); // Temperatur acak antara 23–33
  float humidity = 50 + random(-10, 11); // Kelembapan acak antara 40–60
  int ldrValue = analogRead(LDRPIN) + random(-200, 201);
  float flamePercentage = map(constrain(ldrValue, 0, 4095), 0, 4095, 0, 100);
  int methaneValue = analogRead(METHANE_PIN);
  float methanePercentage = map(constrain(methaneValue, 0, 4095), 0, 4095, 0, 100);
  int toxinValue = analogRead(TOXIN_PIN);
  float toxinPercentage = map(constrain(toxinValue, 0, 4095), 0, 4095, 0, 100);

  // Classify Sensor Values
  String tempClass = classifyTemperature(temperature);
  String humClass = classifyHumidity(humidity);
  String flameClass = classifyFlame(flamePercentage);
  String methaneClass = classifyGas(methanePercentage);
  String toxinClass = classifyGas(toxinPercentage);

  // Control Actuators
  controlActuators(tempClass, humClass, flameClass, methaneClass, toxinClass);

  // Send Data to ThingsBoard
  sendToThingsBoard(temperature, humidity, flamePercentage, methanePercentage, toxinPercentage);

  // Debugging Output
  Serial.printf("Temp: %.2f°C (%s), Hum: %.2f%% (%s), Flame: %.2f%% (%s)\n", temperature, tempClass.c_str(), humidity, humClass.c_str(), flamePercentage, flameClass.c_str());
  Serial.printf("Methane: %.2f%% (%s), Toxin: %.2f%% (%s)\n", methanePercentage, methaneClass.c_str(), toxinPercentage, toxinClass.c_str());
  Serial.println("-----------------------");

  delay(2000);
}

// Function to Setup WiFi
void setupWiFi() {
  delay(10);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
}

// Function to Reconnect MQTT
void reconnect() {
  while (!client.connected()) {
    Serial.println("Attempting MQTT connection...");
    if (client.connect("ESP32Client", token, nullptr)) {
      Serial.println("MQTT connected!");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// Classification Functions
String classifyTemperature(float temperature) {
  if (temperature < 26) return "low";
  else if (temperature <= 40) return "medium";
  else return "high";
}

String classifyHumidity(float humidity) {
  if (humidity > 55) return "low";
  else if (humidity >= 45) return "medium";
  else return "high";
}

String classifyFlame(float flame) {
  if (flame < 5) return "low";
  else if (flame <= 10) return "medium";
  else return "high";
}

String classifyGas(float gas) {
  if (gas < 5) return "low";
  else if (gas <= 10) return "medium";
  else return "high";
}

// Control Actuators
void controlActuators(String tempClass, String humClass, String flameClass, String methaneClass, String toxinClass) {
  digitalWrite(RELAY1, tempClass != "low");
  digitalWrite(RELAY2, humClass != "low");
  digitalWrite(BUZZER, flameClass != "low");
  digitalWrite(RELAY_METHANE, methaneClass != "low");
  digitalWrite(RELAY_TOXIN, toxinClass != "low");
}

// Send Data to ThingsBoard
void sendToThingsBoard(float temperature, float humidity, float flame, float methane, float toxin) {
  String payload = "{\"temperature\":" + String(temperature) +
                   ",\"humidity\":" + String(humidity) +
                   ",\"flame\":" + String(flame) +
                   ",\"methane\":" + String(methane) +
                   ",\"toxin\":" + String(toxin) + "}";
  client.publish("v1/devices/me/telemetry", payload.c_str());
}
