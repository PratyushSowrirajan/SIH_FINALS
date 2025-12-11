# Quick Start Guide - Sign Language Glove

## 1. Configure WiFi (2 minutes)

Edit `src/main.cpp` lines 20-21:
```cpp
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
```

## 2. Build & Upload (3-5 minutes)

```bash
# Build
python -m platformio run

# Upload to ESP32
python -m platformio run --target upload

# Monitor
python -m platformio device monitor
```

## 3. Test Connection (1 minute)

Watch serial monitor for:
```
✓ WiFi connected!
IP Address: 192.168.1.XXX
✓ Web server started
```

You should hear 2 startup beeps!

## 4. Open Web Interface

In browser, go to the IP address shown:
```
http://192.168.1.XXX
```

## 5. Test Everything

### Test Audio (30 seconds)
Click buttons: "Play Hello", "Play Hi", "Test Beep"
→ Should hear tones from speaker

### Test Flex Sensors (1 minute)
Bend each finger, watch values change on web page
- Thumb → Sensor 1
- Index → Sensor 2  
- Middle → Sensor 3
- Ring → Sensor 4
- Pinky → Sensor 5

### Test MPU6050 (30 seconds)
Rotate/move glove, watch gyro X/Y/Z values change

### Test Data Streaming (1 minute)
1. Click "Start Streaming"
2. Open serial monitor
3. Should see CSV data streaming
4. Click "Stop Streaming"

## Done! ✓

Total time: ~10 minutes

## Common Issues

**No WiFi connection?**
- Check SSID/password
- ESP32 only works with 2.4GHz WiFi

**No audio?**
- Check Pin 25 connection
- Verify PAM8403 has 5V from booster
- Test with serial command 't'

**Flex sensors show 0 or 4095?**
- Check wiring
- Need voltage divider circuit
- Pins 34-39 are input-only

**Can't upload code?**
- Hold BOOT button on ESP32
- Select correct COM port
- Try different USB cable
