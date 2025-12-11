"""
Collect sign dataset (time-series method)
For signs like: sample, L, O, C, B, T, good_morning, good_afternoon, good_evening
CSV Format: trial_id,sign_name,time_ms,flex1,flex2,flex3,flex4,flex5,ax,ay,az,gx,gy,gz
"""

import serial
import time
import csv
from datetime import datetime

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
DURATION_SECONDS = 3.0  # Record for 3 seconds per trial (static signs)

print("=" * 70)
print(" SIGN DATASET COLLECTOR (Time-Series)")
print("=" * 70)
print(f"\n⚙️  CONFIGURATION:")
print(f"   - Sampling Rate: ~100 Hz")
print(f"   - Duration per trial: {DURATION_SECONDS} seconds")
print(f"   - Expected samples per trial: ~{int(DURATION_SECONDS * 100)}")
print(f"   - CSV Format: trial_id,sign_name,time_ms,flex1-5,ax,ay,az,gx,gy,gz")
print("\n🤚 SIGNS TO COLLECT:")
print("   - sample (open palm → closed fist)")
print("   - L, O, C, B, T (letter signs)")
print("   - good_morning, good_afternoon, good_evening (dynamic signs)")
print("\n" + "=" * 70)

# Get collection parameters
sign_name = input("\n👉 Sign name (e.g., 'sample', 'L', 'O'): ").strip()
start_trial = int(input(f"👉 Starting trial number (e.g., 1): ").strip())
num_trials = int(input(f"👉 How many '{sign_name}' trials to collect? (recommended: 30-50): ").strip())

print(f"\n✅ Will collect {num_trials} trials of '{sign_name}' starting from trial {start_trial}")
print(f"📁 Output file: {sign_name}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

input("\n⚠️  Press Enter to connect to ESP32...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Connected to ESP32 on COM5\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Prepare CSV file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{sign_name}_dataset_{timestamp}.csv"
csv_file = open(filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['trial_id', 'sign_name', 'time_ms', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

print("=" * 70)
print(" READY TO COLLECT SIGN DATA")
print("=" * 70)

for trial_num in range(start_trial, start_trial + num_trials):
    print(f"\n{'='*70}")
    print(f" TRIAL {trial_num}/{start_trial + num_trials - 1} - {sign_name.upper()}")
    print(f"{'='*70}")
    
    if sign_name.lower() == "sample":
        print("🤚 INSTRUCTION: Open palm → Close fist → Open palm (repeat)")
    else:
        print(f"🤚 INSTRUCTION: Make the '{sign_name}' sign and HOLD IT")
    
    print(f"💡 Duration: {DURATION_SECONDS} seconds")
    
    input(f"\n▶️  Press Enter to START recording trial {trial_num}...")
    
    # Start data collection
    ser.write(b's')
    time.sleep(0.2)
    
    print(f"🔴 RECORDING for {DURATION_SECONDS} seconds...")
    if sign_name.lower() == "sample":
        print("   ✊ OPEN → CLOSE → OPEN your fist repeatedly!")
    else:
        print(f"   🤚 HOLD the '{sign_name}' sign steady!")
    
    trial_data = []
    start_time = time.time()
    first_timestamp = None
    
    while time.time() - start_time < DURATION_SECONDS:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    parts = line.split(',')
                    if len(parts) == 12:  # Time,flex1-5,ax,ay,az,gx,gy,gz
                        timestamp_ms = float(parts[0])
                        
                        if first_timestamp is None:
                            first_timestamp = timestamp_ms
                        
                        relative_time = timestamp_ms - first_timestamp
                        
                        flex1 = int(parts[1])
                        flex2 = int(parts[2])
                        flex3 = int(parts[3])
                        flex4 = int(parts[4])
                        flex5 = int(parts[5])
                        ax = float(parts[6])
                        ay = float(parts[7])
                        az = float(parts[8])
                        gx = float(parts[9])
                        gy = float(parts[10])
                        gz = float(parts[11])
                        
                        trial_data.append([trial_num, sign_name, relative_time, flex1, flex2, flex3, flex4, flex5, ax, ay, az, gx, gy, gz])
                        
            except (UnicodeDecodeError, ValueError, IndexError):
                pass
    
    # Stop collection
    ser.write(b'x')
    time.sleep(0.2)
    
    # Write trial data
    csv_writer.writerows(trial_data)
    csv_file.flush()
    
    print(f"✅ Trial {trial_num} complete: {len(trial_data)} samples collected")
    print(f"   Actual sampling rate: ~{len(trial_data) / DURATION_SECONDS:.1f} Hz")
    
    if trial_num < start_trial + num_trials - 1:
        print("\n⏸️  Get ready for next trial...")
        time.sleep(1)

print("\n" + "=" * 70)
print(" COLLECTION COMPLETE!")
print("=" * 70)
print(f"✅ Collected {num_trials} trials")
print(f"💾 Saved to: {filename}")
print(f"📊 Total rows: approximately {num_trials * 300}")
print("\n🎉 Dataset ready for ML training!")
print("=" * 70)

csv_file.close()
ser.close()
