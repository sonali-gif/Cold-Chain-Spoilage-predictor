#include <DHT.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <HTTPClient.h>

// ---------- Pin setup ----------
#define DHTPIN 4
#define DHTTYPE DHT22
#define GAS_PIN 34

DHT dht(DHTPIN, DHTTYPE);
WiFiMulti wifiMulti;

// ---------- Google Sheets ----------
const String scriptURL = "https://script.google.com/macros/s/AKfycbyeKH5wwu_sySA6MKOl4Fy_3DI-KGsVcdjYCHt0LbIveI2h7_7kOo2DBWdNy1FEAflFLw/exec";

// ---------- Calibration ----------
float R0 = 0;
const int CALIBRATION_SAMPLES = 50;
const unsigned long SAMPLE_INTERVAL = 5UL * 60UL * 1000UL; // 5 minutes
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  dht.begin();

  // Add as many networks as you want here
  wifiMulti.addAP("Nothing Phone 2", "Sparsh@111");
  wifiMulti.addAP("Airtel_Sparsh", "Airtel@123");
  // wifiMulti.addAP("ANOTHER_NETWORK", "ANOTHER_PASSWORD");

  Serial.print("Connecting to Wi-Fi");
  while (wifiMulti.run() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("Connected to: ");
  Serial.println(WiFi.SSID());

  Serial.println("Calibrating gas sensor baseline (R0)...");
  calibrateR0();
  Serial.print("R0 calibrated: ");
  Serial.println(R0);
}

void loop() {
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL || lastSampleTime == 0) {
    lastSampleTime = millis();
    takeReadingAndSend();
  }
}

void calibrateR0() {
  float sum = 0;
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    sum += getRs();
    delay(100);
  }
  R0 = sum / CALIBRATION_SAMPLES;
}

float getRs() {
  int raw = analogRead(GAS_PIN);
  float voltage = raw * (3.3 / 4095.0);
  if (voltage == 0) voltage = 0.01;
  float Rs = (3.3 - voltage) / voltage;
  return Rs;
}

void takeReadingAndSend() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  float Rs = getRs();
  float ratio = (R0 != 0) ? (Rs / R0) : 0;

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT22!");
    return;
  }

  Serial.println("---- New Reading ----");
  Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" C");
  Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
  Serial.print("Rs/R0 ratio: "); Serial.println(ratio);

  sendToGoogleSheets(temperature, humidity, ratio);
}

void sendToGoogleSheets(float temp, float hum, float ratio) {
  if (wifiMulti.run() != WL_CONNECTED) {
    Serial.println("Wi-Fi not connected, skipping upload.");
    return;
  }

  HTTPClient http;
  String url = scriptURL + "?temp=" + String(temp) +
               "&hum=" + String(hum) +
               "&ratio=" + String(ratio);

  http.begin(url);
  int httpCode = http.GET();

  if (httpCode > 0) {
    Serial.print("Data sent, response code: ");
    Serial.println(httpCode);
  } else {
    Serial.print("Error sending data: ");
    Serial.println(http.errorToString(httpCode));
  }
  http.end();
}