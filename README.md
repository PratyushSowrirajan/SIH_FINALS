# Sign Language Recognition System - Technical Documentation

## Project Context for AI Assistant

This is a **sign language recognition system** using ESP32 microcontroller with MPU6050 IMU sensor and flex sensors for hand gesture detection. The project was developed incrementally with calibration, data collection, and machine learning components.

---

## Hardware Setup

### Components
- **ESP32-D0WD-V3** microcontroller (revision v3.1)
- **MPU6050** IMU sensor (I2C: SDA=21, SCL=22)
- **5 Flex Sensors** (currently simulated with random data: 200-1300 range)
  - 200 = no bend (straight)
  - 1300 = 90° bend (maximum)
- Serial communication: **COM5** at **115200 baud**

### Sensor Configuration (Optimized for Gesture Recognition)
- Accelerometer: **±2g range** (highest resolution)
- Gyroscope: **±250 deg/s range** (highest resolution)
- DLPF Filter: **~42 Hz** (reduces noise)
- Sampling Rate: **~100 Hz**

### Calibration Status
**MPU6050 is pre-calibrated** with the following offsets (already applied in code):
```cpp
GX_OFFSET = 0.308f;   // deg/s
GY_OFFSET = 0.587f;   // deg/s
GZ_OFFSET = -0.044f;  // deg/s
AX_OFFSET = 0.0247f;  // g
AY_OFFSET = -0.0249f; // g
AZ_OFFSET = 0.0278f;  // g
```
**Do NOT recalibrate** unless sensor is replaced or readings are incorrect.

---

## Project Structure

### ESP32 Code
- **File:** `src/main.cpp`
- **Framework:** Arduino (PlatformIO)
- **Platform:** espressif32 @ 6.12.0
- **Output Format:** `Time(ms),flex1,flex2,flex3,flex4,flex5,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps`
- **Commands:**
  - Send `s` or `S` → Start data collection
  - Send `x` or `X` → Stop data collection

### Python Scripts

#### Data Collection
1. **`collect_signs.py`** - Universal sign data collector
   - Time-series method (3 seconds per trial)
   - Works for all sign types (static letters, dynamic gestures)
   - Output: `trial_id,sign_name,time_ms,flex1-5,ax,ay,az,gx,gy,gz`
   - Usage: `python collect_signs.py`

2. **`collect_hi_dataset.py`** - Original HI gesture collector
   - 2-second recordings for simple gestures
   - Format: `trial_id,label,time_ms,ax,ay,az,gx,gy,gz` (no flex)

#### Calibration & Testing (Legacy - from early development)
- **`calibrate_mpu.py`** - Automatic MPU6050 calibration (500 samples)
- **`collect_verification.py`** - Post-calibration verification
- **`analyze_calibration.py`** - Calibration quality analysis
- **`axis_sanity_check.py`** - Real-time axis monitoring
- **`compare_crc_movements.py`** - Gesture repeatability testing

#### Model Training & Testing
1. **`train_hi_model.py`** - Train HI gesture detector
   - Template-based (Euclidean distance)
   - Requires: `hi_dataset_*.csv` and `not_hi_dataset_*.csv`
   - Outputs: `hi_gesture_model.pkl`

2. **`test_hi_realtime.py`** - Real-time HI gesture detection
   - Loads `hi_gesture_model.pkl`
   - Records 2 seconds and classifies as HI or NOT_HI

3. **`train_sample_model.py`** - Train SAMPLE gesture detector (if exists)
4. **`test_sample_realtime.py`** - Test SAMPLE gesture (if exists)

#### Visualization
- **`visualize_realtime.py`** - Live matplotlib graphs of sensor data
  - Shows all 6 axes (accel + gyro) in real-time

---

## Datasets Collected

### Status: ✅ Collected | ⏳ Pending

| Sign Name | File | Trials | Status | Description |
|-----------|------|--------|--------|-------------|
| **hi** | `hi_dataset_20251210_030854.csv` | 40 | ✅ | Center → Left → Center wave |
| **not_hi** | `not_hi_dataset_20251210_031340.csv` | 40 | ✅ | Random/still movements (negative examples) |
| **sample** | `sample_dataset_20251210_111532.csv` | 40 | ✅ | Open palm → Closed fist (repeated) |
| **hello** | `hello_dataset_20251210_112916.csv` | 40 | ✅ | Hello gesture |
| **eight** | `eight_dataset_20251210_113906.csv` | 40 | ✅ | Number 8 sign |
| **L** | - | - | ⏳ | Letter L sign |
| **O** | - | - | ⏳ | Letter O sign |
| **C** | - | - | ⏳ | Letter C sign |
| **B** | - | - | ⏳ | Letter B sign |
| **T** | - | - | ⏳ | Letter T sign |
| **good_morning** | - | - | ⏳ | Dynamic greeting gesture |
| **good_afternoon** | - | - | ⏳ | Dynamic greeting gesture |
| **good_evening** | - | - | ⏳ | Dynamic greeting gesture |

### Dataset Format
All datasets use **time-series format** with columns:
```
trial_id, sign_name, time_ms, flex1, flex2, flex3, flex4, flex5, ax, ay, az, gx, gy, gz
```

---

## Trained Models

### HI Gesture Detector
- **File:** `hi_gesture_model.pkl`
- **Algorithm:** Template matching (Euclidean distance)
- **Training Accuracy:** 81.2%
- **Training Date:** 2025-12-10 03:20:31
- **Threshold:** 1161.78
- **Method:** 
  - Averages all HI gesture trials → creates template
  - New gestures compared to template
  - Distance < threshold → HI detected
  - Distance ≥ threshold → NOT HI

---

## Development History & Key Decisions

### Phase 1: MPU6050 Setup & Calibration
1. ✅ Installed PlatformIO IDE + C/C++ extensions
2. ✅ Created ESP32 project with MPU6050 code
3. ✅ Fixed upload issues (COM port conflicts with Arduino IDE)
4. ✅ Removed Madgwick filter (simplified to raw accel + gyro only)
5. ✅ Performed automatic calibration (500 samples, 5 seconds)
6. ✅ Applied calibration offsets to code
7. ✅ Verified calibration quality (0.982g gravity magnitude - excellent)
8. ✅ Configured sensor ranges (±2g accel, ±250 deg/s gyro, DLPF ~42Hz)

### Phase 2: Data Collection System
1. ✅ Added start/stop controls (`s`/`x` commands)
2. ✅ Created Python CSV capture scripts
3. ✅ Implemented axis sanity checks (movement → axis mapping)
4. ✅ Tested gesture repeatability (Center→Right→Center had <20% variance)

### Phase 3: Flex Sensor Integration
1. ✅ Added 5 flex sensor columns to ESP32 code
2. ✅ Implemented random data generation (200-1300) as placeholder
3. ✅ Updated CSV format to include flex readings
4. ✅ Note: **Real flex sensors not yet connected** - currently simulated

### Phase 4: Sign Language Dataset Collection
1. ✅ Created unified `collect_signs.py` for all sign types
2. ✅ Collected initial datasets: sample, hello, eight (40 trials each)
3. ✅ Collected HI gesture training data (40 HI + 40 NOT_HI)
4. ⏳ Pending: Letter signs (L, O, C, B, T) and dynamic greetings

### Phase 5: Machine Learning
1. ✅ Built template-based HI gesture classifier
2. ✅ Trained model with 81.2% accuracy
3. ✅ Created real-time detection system
4. ⏳ Pending: Multi-sign classifier for all collected signs

---

## Workflow Guide for AI Assistant

### To Collect New Sign Data
```bash
python collect_signs.py
# Enter sign name (e.g., "L", "good_morning")
# Enter starting trial number (usually 1)
# Enter number of trials (30-50 recommended)
# Perform gesture during each 3-second recording
```

### To Upload Code to ESP32
```bash
& "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run --target upload
```

### To Train a New Model
1. Ensure datasets exist (e.g., `L_dataset_*.csv`)
2. Create training script based on `train_hi_model.py` template
3. Template matching approach:
   - Load all trials for each sign
   - Create average template per sign
   - Calculate distance thresholds
   - Save model as `.pkl` file

### To Test a Model in Real-Time
1. Load trained model (`.pkl` file)
2. Connect to ESP32 (COM5, 115200 baud)
3. Send `s` command to start collection
4. Record 2-3 seconds of data
5. Resample to match training length
6. Calculate distance to all sign templates
7. Classify based on minimum distance

---

## Important Technical Notes

### MPU6050 Axis Mapping (User's Hardware)
From axis sanity checks:
- **Accel X** = Left/Right movement (changes ~0.75g during side motion)
- **Accel Z** = Up/Down (shows ~1g when still due to gravity pointing down)
- **Gyro Y** = Rotation around Y-axis (most consistent for gestures)

### Sensor Drift Characteristics
- **Gyro Y drift:** 1.78 deg/s (acceptable, within normal range)
- **Accel gravity magnitude:** 0.982g (excellent, close to ideal 1.0g)
- **Calibration quality:** Excellent - no need to recalibrate

### Serial Communication Protocol
- **Baud rate:** 115200
- **Line ending:** CRLF (Windows-style)
- **Data format:** CSV with 12 columns (time + 5 flex + 6 IMU)
- **Timing:** ~10ms delay between samples (~100Hz)

### Flex Sensor Status
**CRITICAL:** Flex sensors are currently **simulated with random data**!
- Code generates random values: `random(200, 1301)`
- When real sensors are connected:
  1. Replace `random()` calls with actual analog reads
  2. Map analog values (0-4095 for ESP32) to calibrated ranges
  3. May need individual calibration per finger

---

## Common Issues & Solutions

### Issue: COM Port Not Found
- **Symptom:** `serial.Serial(SERIAL_PORT, BAUD_RATE)` fails
- **Solution:** Check Device Manager → Ports (COM & LPT) → Find ESP32 port number
- **Fix:** Update `SERIAL_PORT = 'COM5'` in Python scripts to correct port

### Issue: Upload Failed / Port Busy
- **Symptom:** PlatformIO upload fails with "port busy"
- **Solution:** 
  1. Close all serial monitor terminals
  2. Stop all Python scripts using serial port
  3. Unplug/replug ESP32 USB
  4. Retry upload

### Issue: Compiler Warnings (Wire.requestFrom)
- **Symptom:** Ambiguous function warnings during compilation
- **Status:** Harmless - ESP32 Arduino framework issue
- **Action:** Ignore warnings - code still works correctly

### Issue: Model Not Found
- **Symptom:** `FileNotFoundError: hi_gesture_model.pkl`
- **Solution:** Run `python train_hi_model.py` first to generate model

### Issue: Python Module Not Found
- **Symptom:** `ModuleNotFoundError: No module named 'pandas'` (or numpy, pyserial)
- **Solution:** `pip install pandas numpy pyserial`

---

## Next Steps & Future Development

### Immediate Tasks
1. ⏳ Collect remaining sign datasets (L, O, C, B, T)
2. ⏳ Collect dynamic gesture datasets (good_morning, good_afternoon, good_evening)
3. ⏳ Train multi-class sign classifier (all signs)
4. ⏳ Build unified real-time recognition system

### Hardware Upgrades
1. ⏳ Connect real flex sensors (replace random data)
2. ⏳ Calibrate flex sensors (min/max bending values)
3. ⏳ Add second hand support (another MPU6050 + 5 flex sensors)
4. ⏳ Consider emergency signs dataset

### Model Improvements
1. ⏳ Try machine learning approaches:
   - Random Forest classifier
   - 1D CNN (Convolutional Neural Network)
   - LSTM (Long Short-Term Memory) for time series
2. ⏳ Implement DTW (Dynamic Time Warping) for better time-series matching
3. ⏳ Cross-validation and hyperparameter tuning
4. ⏳ Confusion matrix analysis for multi-class classification

---

## Repository Information

- **GitHub:** https://github.com/RoshanJ007/deliverables
- **Branch:** main
- **Last Commit:** Initial commit - Sign language recognition with ESP32 and MPU6050
- **Files:** 33 files, 36,086+ lines

### Git Workflow
```bash
# To update repository after changes
git add .
git commit -m "Your commit message"
git push origin main

# To check status
git status

# To view history
git log --oneline
```

---

## Quick Reference Commands

### ESP32 Development
```bash
# Upload code to ESP32
platformio run --target upload

# Monitor serial output
platformio device monitor
```

### Data Collection
```bash
# Collect any sign dataset
python collect_signs.py

# Real-time visualization
python visualize_realtime.py
```

### Model Training & Testing
```bash
# Train HI gesture model
python train_hi_model.py

# Test HI gesture in real-time
python test_hi_realtime.py
```

### Calibration (if needed)
```bash
# Recalibrate MPU6050 (use only if necessary)
python calibrate_mpu.py

# Verify calibration quality
python collect_verification.py
python analyze_calibration.py
```

---

## Contact & Maintenance

**Project Owner:** RoshanJ007  
**Development Platform:** VS Code + PlatformIO + Python 3.14  
**Last Updated:** December 10-11, 2025

**For AI Assistant:**
- All calibration data is preserved in code
- Model files contain training metadata
- Dataset filenames include timestamps for version tracking
- Follow time-series approach for consistency
- Always verify COM port on new systems
- Flex sensor data is currently simulated - remember this when analyzing results

---

*This README serves as technical documentation for both human developers and AI assistants to understand the project state, decisions made, and continuation points.*
