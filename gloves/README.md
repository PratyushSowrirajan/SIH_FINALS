# Sign Language Glove - Wireless Test System

## Hardware Setup

### Components Connected
- **ESP32 DevKit** - Main controller
- **5 Flex Sensors** - Finger bend detection
- **MPU6050** - 6-axis motion sensor
- **PAM8403** - Audio amplifier
- **Speaker** - Audio output
- **LiPo Battery** - 3.7V power source
- **SS24 Voltage Booster** - 5V for audio amplifier

### Pin Connections

#### Flex Sensors (Analog Input)
- Pin 32 → Flex Sensor 1 (Thumb)
- Pin 33 → Flex Sensor 2 (Index)
- Pin 34 → Flex Sensor 3 (Middle)
- Pin 35 → Flex Sensor 4 (Ring)
- Pin 36 → Flex Sensor 5 (Pinky)

#### MPU6050 (I2C)
- Pin 21 → SDA
- Pin 22 → SCL
- 3.3V → VCC
- GND → GND

#### PAM8403 Audio Amplifier
- Pin 25 → PAM8403 L Input (PWM Audio)
- 5V (from SS24 booster) → PAM8403 VCC
- GND → GND
- Speaker → PAM8403 L Output

#### Power
- LiPo 3.7V → ESP32 VIN (or USB)
- LiPo 3.7V → SS24 Input
- SS24 5V Output → PAM8403 VCC

## Features

### 1. Real-time Sensor Monitoring
- Reads all 5 flex sensors at ~100Hz
- MPU6050 accelerometer and gyroscope data
- Calibrated offsets applied automatically

### 2. WiFi Web Interface
- Access via browser: `http://[ESP32-IP-ADDRESS]`
- Live sensor data updates
- Visual display of all sensors
- Control buttons for audio playback

### 3. Audio Playback
- Plays different tone patterns for words:
  - "Hello" - 4-tone pattern
  - "Hi" - 2-tone pattern  
  - "Thank You" - 5-tone pattern
  - "Yes" - 2 ascending tones
  - "No" - 2 descending tones
  - "Test" - Single beep

### 4. Data Streaming
- Serial output at 115200 baud
- CSV format for easy data collection
- Start/Stop control via serial or web

## Configuration

### WiFi Settings
Before uploading, edit `src/main.cpp` lines 20-21:

```cpp
const char* ssid = "YourWiFiSSID";        // Change this
const char* password = "YourWiFiPassword"; // Change this
```

## Building and Uploading

### Using PlatformIO CLI
```bash
# Build the project
python -m platformio run

# Upload to ESP32
python -m platformio run --target upload

# Monitor serial output
python -m platformio device monitor
```

### Using VS Code
1. Open folder in VS Code
2. Install PlatformIO extension
3. Click "Build" button (✓)
4. Click "Upload" button (→)
5. Click "Serial Monitor" button

## Usage

### Serial Commands
Connect via serial monitor at 115200 baud:
- `s` - Start data streaming
- `x` - Stop data streaming
- `h` - Play "Hello" sound
- `t` - Test audio tone

### Web Interface
1. Upload code and open serial monitor
2. Note the IP address displayed (e.g., `192.168.1.100`)
3. Open browser and go to: `http://192.168.1.100`
4. Use buttons to:
   - View live sensor data
   - Test audio playback
   - Control data streaming

## Testing Procedure

1. **Power On Test**
   - Connect USB or battery
   - Should hear 2 startup beeps
   - Check serial monitor for WiFi connection

2. **Flex Sensor Test**
   - Open web interface
   - Bend each finger
   - Watch corresponding sensor value change
   - Values should be: ~0-4095 (12-bit ADC)

3. **MPU6050 Test**
   - Keep glove still - gyro values near 0
   - Rotate glove - see gyro values change
   - Expected range: ±250 deg/s

4. **Audio Test**
   - Click each "Play" button on web interface
   - Or send 'h' or 't' via serial
   - Should hear tones from speaker

5. **Data Collection Test**
   - Start streaming (web or serial 's')
   - Perform hand gestures
   - Stop streaming ('x')
   - Save serial data for training

## Troubleshooting

### WiFi Not Connecting
- Check SSID and password
- ESP32 only supports 2.4GHz WiFi
- Check router settings

### No Audio Output
- Verify PAM8403 has 5V power
- Check speaker connections
- Test with serial command 't'
- Check Pin 25 connection

### Flex Sensors Not Reading
- Pins 34-39 are input-only (no pull-up)
- Use voltage divider circuit
- Check ADC readings in serial monitor

### MPU6050 Not Working
- Check I2C connections (SDA/SCL)
- Verify 3.3V power
- Try I2C scanner code

## Data Format

### Serial CSV Output
```
Time(ms),flex1,flex2,flex3,flex4,flex5,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
1234,2048,1890,2200,1950,2100,-0.0123,0.0456,0.9876,1.234,-0.567,0.890
```

### Web JSON Output
```json
{
  "flex1": 2048,
  "flex2": 1890,
  "flex3": 2200,
  "flex4": 1950,
  "flex5": 2100,
  "gx": 1.23,
  "gy": -0.56,
  "gz": 0.89
}
```

## Next Steps

1. **Calibrate Flex Sensors**
   - Record min/max values for each finger
   - Add normalization in code

2. **Train ML Models**
   - Collect gesture datasets
   - Train with existing Python scripts
   - Deploy model to ESP32

3. **Add More Audio**
   - Record/generate more words
   - Add text-to-speech library
   - Store audio in SPIFFS/SD

4. **Improve UI**
   - Add charts and graphs
   - Historical data display
   - Gesture recognition display

## Safety Notes

⚠️ **Important**
- LiPo batteries can be dangerous if mishandled
- Never short circuit battery terminals
- Use proper LiPo charger
- Monitor battery voltage (don't discharge below 3.0V)
- Keep battery away from heat/fire
- SS24 booster gets warm - ensure ventilation

## Support

For issues or questions, check:
- Serial monitor output for error messages
- WiFi connection status
- Component wiring
- Power supply voltage levels
