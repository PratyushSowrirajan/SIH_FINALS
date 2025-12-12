/* ESP32 MPU6050 -> Calibrated + Madgwick fusion + EEPROM
   Pins: SDA=21, SCL=22
   Serial: 115200
   Make sure to keep MPU still & flat on first run.
   Madgwick implementation below (lightweight).
*/

#include <Wire.h>
#include <EEPROM.h>
#include <math.h>

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

// EEPROM addresses
const int ADDR_SIGN = 0;          // signature
const int ADDR_OFFS = 4;          // offsets start

// Data collection control
bool collecting = false;

// Forward declarations
void readRawSigned(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz);

void writeInt16ToEEPROM(int addr, int16_t v) {
  EEPROM.write(addr, (v >> 8) & 0xFF);
  EEPROM.write(addr+1, v & 0xFF);
}
int16_t readInt16FromEEPROM(int addr) {
  uint8_t hi = EEPROM.read(addr);
  uint8_t lo = EEPROM.read(addr+1);
  return (int16_t)((hi << 8) | lo);
}



void setup() {
  Serial.begin(115200);
  Wire.begin(21,22);
  delay(200);

  // wake MPU
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(100);

  // Configure accelerometer range: ±2g (highest resolution)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG register
  Wire.write(0x00);  // ±2g (00 = ±2g, 08 = ±4g, 10 = ±8g, 18 = ±16g)
  Wire.endTransmission(true);

  // Configure gyroscope range: ±250 deg/s (highest resolution)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG register
  Wire.write(0x00);  // ±250 deg/s (00 = ±250, 08 = ±500, 10 = ±1000, 18 = ±2000)
  Wire.endTransmission(true);

  // Configure DLPF (Digital Low Pass Filter) ~42 Hz
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A);  // CONFIG register
  Wire.write(0x03);  // DLPF_CFG = 3 (~44 Hz accel, ~42 Hz gyro)
  Wire.endTransmission(true);

  delay(100);

  Serial.println("MPU6050 initialized with calibrated offsets.");
  Serial.println("Settings: ±2g accel, ±250 deg/s gyro, DLPF ~42Hz");

  // Seed random number generator for flex sensor simulation
  randomSeed(analogRead(0));

  Serial.println("\n=== MPU6050 + Flex Sensors Data Logger ===");
  Serial.println("Commands:");
  Serial.println("  's' - Start data collection");
  Serial.println("  'x' - Stop data collection");
  Serial.println("Note: Flex sensors are SIMULATED (random values)");
  Serial.println("==========================================\n");
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') {
      collecting = true;
      Serial.println("\n>>> DATA COLLECTION STARTED <<<");
      Serial.println("Time(ms),flex1,flex2,flex3,flex4,flex5,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
    } else if (cmd == 'x' || cmd == 'X') {
      collecting = false;
      Serial.println(">>> DATA COLLECTION STOPPED <<<\n");
      Serial.println("Send 's' to start collecting again.");
    }
  }

  int16_t axr, ayr, azr, gxr, gyr, gzr;
  readRawSigned(axr, ayr, azr, gxr, gyr, gzr);

  // convert to physical units
  float ax_g = (float)axr / ACCEL_SENS;
  float ay_g = (float)ayr / ACCEL_SENS;
  float az_g = (float)azr / ACCEL_SENS;

  float gx_dps = (float)gxr / GYRO_SENS;
  float gy_dps = (float)gyr / GYRO_SENS;
  float gz_dps = (float)gzr / GYRO_SENS;

  // apply calibrated offsets
  ax_g -= AX_OFFSET;
  ay_g -= AY_OFFSET;
  az_g -= AZ_OFFSET;
  gx_dps -= GX_OFFSET;
  gy_dps -= GY_OFFSET;
  gz_dps -= GZ_OFFSET;

  unsigned long now = millis();

  // Generate random flex sensor values (simulated)
  // Range: 300 (no bend) to 1800 (full bend)
  int flex1 = random(300, 1801);
  int flex2 = random(300, 1801);
  int flex3 = random(300, 1801);
  int flex4 = random(300, 1801);
  int flex5 = random(300, 1801);

  // print CSV line only if collecting
  if (collecting) {
    Serial.print(now); Serial.print(",");
    Serial.print(flex1); Serial.print(",");
    Serial.print(flex2); Serial.print(",");
    Serial.print(flex3); Serial.print(",");
    Serial.print(flex4); Serial.print(",");
    Serial.print(flex5); Serial.print(",");
    Serial.print(ax_g,4); Serial.print(",");
    Serial.print(ay_g,4); Serial.print(",");
    Serial.print(az_g,4); Serial.print(",");
    Serial.print(gx_dps,3); Serial.print(",");
    Serial.print(gy_dps,3); Serial.print(",");
    Serial.println(gz_dps,3);
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
