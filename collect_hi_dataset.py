"""
Collect "HI" gesture dataset for ML training
Collects trials with format: trial_id,label,time_ms,ax,ay,az,gx,gy,gz
"""

import serial
import time
import csv
from datetime import datetime

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
DURATION_SECONDS = 2.0  # Record for 2 seconds per trial
TARGET_HZ = 100  # Target sampling rate

print("=" * 70)
print(" 'HI' GESTURE DATASET COLLECTOR")
print("=" * 70)
print(f"\n⚙️  CONFIGURATION:")
print(f"   - Sampling Rate: ~{TARGET_HZ} Hz")
print(f"   - Duration per trial: {DURATION_SECONDS} seconds")
print(f"   - Expected samples per trial: ~{int(DURATION_SECONDS * TARGET_HZ)}")
print(f"   - CSV Format: trial_id,label,time_ms,ax,ay,az,gx,gy,gz")
print("\n🎯 GESTURE: CENTER → LEFT → CENTER (hi wave)")
print("\n📝 LABELS:")
print("   1 = 'hi' gesture")
print("   0 = 'not_hi' (still/random/other)")
print("\n" + "=" * 70)

# Get collection parameters
label_choice = input("\n👉 Collecting (1) HI gestures or (0) NOT_HI gestures? Enter 1 or 0: ").strip()
label = int(label_choice)
label_name = "HI" if label == 1 else "NOT_HI"

start_trial = int(input(f"👉 Starting trial number (e.g., 1): ").strip())
num_trials = int(input(f"👉 How many {label_name} trials to collect? (recommended: 30-50): ").strip())

print(f"\n✅ Will collect {num_trials} trials of '{label_name}' starting from trial {start_trial}")
print(f"📁 Output file: {label_name.lower()}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

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
filename = f"{label_name.lower()}_dataset_{timestamp}.csv"
csv_file = open(filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['trial_id', 'label', 'time_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

print("=" * 70)
print(" READY TO COLLECT DATA")
print("=" * 70)

for trial_num in range(start_trial, start_trial + num_trials):
    print(f"\n{'='*70}")
    print(f" TRIAL {trial_num}/{start_trial + num_trials - 1} - {label_name}")
    print(f"{'='*70}")
    
    if label == 1:
        print("🎯 Prepare: Position hand at CENTER")
        print("💡 Motion: CENTER → LEFT → CENTER (hi wave)")
    else:
        print("🎯 Prepare: Ready for NOT_HI motion")
        print("💡 Options: Keep hand still / random movement / other gesture")
    
    input(f"\n▶️  Press Enter to START recording trial {trial_num}...")
    
    # Start data collection
    ser.write(b's')
    time.sleep(0.2)
    
    print(f"🔴 RECORDING for {DURATION_SECONDS} seconds...")
    if label == 1:
        print("   👋 DO THE HI GESTURE NOW!")
    
    trial_data = []
    start_time = time.time()
    first_timestamp = None
    
    while time.time() - start_time < DURATION_SECONDS:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    parts = line.split(',')
                    if len(parts) == 7:
                        timestamp_ms = float(parts[0])
                        
                        if first_timestamp is None:
                            first_timestamp = timestamp_ms
                        
                        relative_time = timestamp_ms - first_timestamp
                        
                        ax = float(parts[1])
                        ay = float(parts[2])
                        az = float(parts[3])
                        gx = float(parts[4])
                        gy = float(parts[5])
                        gz = float(parts[6])
                        
                        trial_data.append([trial_num, label, relative_time, ax, ay, az, gx, gy, gz])
                        
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
print(f"📊 Total rows: {(trial_num - start_trial + 1) * len(trial_data)}")
print("\n🎉 Dataset ready for ML training!")
print("=" * 70)

csv_file.close()
ser.close()
