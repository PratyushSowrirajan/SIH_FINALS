/* ESP32 LEFT GLOVE - WiFi Version
   Same MPU6050 + Fake Flex functionality as USB version
   Sends data over WiFi TCP connection instead of Serial
   Connect to this ESP32's WiFi network, then TCP to port 8080
*/

#include <Wire.h>
#include <WiFi.h>
#include <math.h>

// WiFi Configuration (ESP32 connects to phone hotspot)
const char* ssid = "Roshan";
const char* password = "12345678";
WiFiServer server(8080);
WiFiClient client;

const int MPU_ADDR = 0x68;
const float GYRO_SENS = 131.0f;   // LSB/(deg/s) for ±250dps
const float ACCEL_SENS = 16384.0f; // LSB/g for ±2g

// ----- Calibrated offsets (auto-generated) -----
float GX_OFFSET = 0.308f;   // deg/s
float GY_OFFSET = 0.587f;   // deg/s
float GZ_OFFSET = -0.044f;  // deg/s
float AX_OFFSET = 0.0247f;  // g
float AY_OFFSET = -0.0249f; // g
float AZ_OFFSET = 0.0278f;  // g
// -----------------------------------------------

// Data collection control
bool collecting = false;

// Forward declarations
void readRawSigned(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz);

void setup() {
  Serial.begin(115200);
  Wire.begin(21,22);
  delay(200);

  // Connect to WiFi (phone hotspot)
  Serial.println("Connecting to WiFi: Roshan");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    IPAddress IP = WiFi.localIP();
    Serial.print("LEFT GLOVE IP: ");
    Serial.println(IP);
    Serial.println("*** WRITE DOWN THIS IP ADDRESS ***");
    
    server.begin();
    Serial.println("TCP Server started on port 8080");
  } else {
    Serial.println("\nWiFi connection failed!");
  }

  // wake MPU
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(100);

  // Configure accelerometer range: ±2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Configure gyroscope range: ±250 deg/s
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Configure DLPF ~42 Hz
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A);
  Wire.write(0x03);
  Wire.endTransmission(true);

  delay(100);

  Serial.println("MPU6050 initialized - LEFT GLOVE");
  Serial.println("Waiting for WiFi client connection...");

  // Seed random for flex sensors
  randomSeed(analogRead(0));
}

void loop() {
  // Accept new client if not connected
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected!");
      client.println("LEFT_GLOVE_READY");
      client.println("Commands: 's' - Start, 'x' - Stop");
    }
  }

  // Check for commands from WiFi client
  if (client && client.available() > 0) {
    char cmd = client.read();
    if (cmd == 's' || cmd == 'S') {
      collecting = true;
      Serial.println(">>> DATA COLLECTION STARTED <<<");
      client.println(">>> DATA COLLECTION STARTED <<<");
      client.println("Time(ms),flex1,flex2,flex3,flex4,flex5,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
    } else if (cmd == 'x' || cmd == 'X') {
      collecting = false;
      Serial.println(">>> DATA COLLECTION STOPPED <<<");
      client.println(">>> DATA COLLECTION STOPPED <<<");
    }
  }

  // Read MPU data
  int16_t axr, ayr, azr, gxr, gyr, gzr;
  readRawSigned(axr, ayr, azr, gxr, gyr, gzr);

  // Convert to physical units
  float ax_g = (float)axr / ACCEL_SENS;
  float ay_g = (float)ayr / ACCEL_SENS;
  float az_g = (float)azr / ACCEL_SENS;

  float gx_dps = (float)gxr / GYRO_SENS;
  float gy_dps = (float)gyr / GYRO_SENS;
  float gz_dps = (float)gzr / GYRO_SENS;

  // Apply calibrated offsets
  ax_g -= AX_OFFSET;
  ay_g -= AY_OFFSET;
  az_g -= AZ_OFFSET;
  gx_dps -= GX_OFFSET;
  gy_dps -= GY_OFFSET;
  gz_dps -= GZ_OFFSET;

  unsigned long now = millis();

  // Generate random flex sensor values (300-1800)
  int flex1 = random(300, 1801);
  int flex2 = random(300, 1801);
  int flex3 = random(300, 1801);
  int flex4 = random(300, 1801);
  int flex5 = random(300, 1801);

  // Send CSV line if collecting and client connected
  if (collecting && client && client.connected()) {
    String data = String(now) + "," +
                  String(flex1) + "," +
                  String(flex2) + "," +
                  String(flex3) + "," +
                  String(flex4) + "," +
                  String(flex5) + "," +
                  String(ax_g, 4) + "," +
                  String(ay_g, 4) + "," +
                  String(az_g, 4) + "," +
                  String(gx_dps, 3) + "," +
                  String(gy_dps, 3) + "," +
                  String(gz_dps, 3);
    
    client.println(data);
    Serial.println(data);  // Also to serial for debugging
  }

  delay(10); // ~100 Hz
}

void readRawSigned(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  ax = (int16_t)(Wire.read() << 8 | Wire.read());
  ay = (int16_t)(Wire.read() << 8 | Wire.read());
  az = (int16_t)(Wire.read() << 8 | Wire.read());
  Wire.read(); Wire.read(); // temp
  gx = (int16_t)(Wire.read() << 8 | Wire.read());
  gy = (int16_t)(Wire.read() << 8 | Wire.read());
  gz = (int16_t)(Wire.read() << 8 | Wire.read());
}
