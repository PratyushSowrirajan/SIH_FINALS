# DUAL GLOVE SETUP GUIDE

## Current Status:
✅ LEFT glove has WiFi code uploaded (currently connected via USB)
✅ RIGHT glove USB code ready
✅ Dual collection script created

## Next Steps:

### Step 1: Setup LEFT Glove (WiFi)
1. **Disconnect LEFT glove** from USB cable
2. Power it separately (battery/power bank)
3. It will create WiFi network: `LEFT_GLOVE` (password: `glove123`)

### Step 2: Connect RIGHT Glove (USB)
1. **Connect RIGHT glove** to USB cable (COM6)
2. Upload USB code to it:
   ```
   python -m platformio run --target upload --upload-port COM6
   ```

### Step 3: Network Connection
1. On your computer, **connect to WiFi**: `LEFT_GLOVE` (password: `glove123`)
2. LEFT glove will be at IP: `192.168.4.1` port `8080`

### Step 4: Collect Data
1. Run the dual collection script:
   ```
   python collect_dual_gloves.py
   ```
2. Follow prompts (sign name, number of trials)
3. Perform signs with BOTH hands simultaneously
4. Data merges automatically into single CSV

## CSV Format (22 columns):
```
trial_id, sign_name, time_ms,
flex_right_1-5 (5 columns),
flex_left_1-5 (5 columns),
ax_right, ay_right, az_right, gx_right, gy_right, gz_right (6 columns),
ax_left, ay_left, az_left, gx_left, gy_left, gz_left (6 columns)
```

## Files Created:
- `src/main_wifi_left.cpp` - WiFi code for LEFT glove
- `src/main_usb_right.cpp` - USB code for RIGHT glove
- `collect_dual_gloves.py` - Dual glove collection script

## Current State:
- LEFT glove: Has WiFi code, still connected to USB
- RIGHT glove: Needs USB code upload
- Computer: Needs to disconnect from current WiFi and connect to LEFT_GLOVE
