"""
Dual Glove Dataset Collector - RIGHT (COM6) + LEFT (COM7) via USB
Merges data from both gloves into single CSV with 24 columns
"""

import serial
import time
import csv
import threading
from datetime import datetime
from queue import Queue

# Port assignments (verified by unplugging test)
RIGHT_GLOVE_PORT = 'COM6'
LEFT_GLOVE_PORT = 'COM8'
BAUD_RATE = 115200
DURATION = 3.0  # seconds per trial

# Calibration offsets (from calibration_offsets.txt)
CAL_GX = 0.308
CAL_GY = 0.587
CAL_GZ = -0.044
CAL_AX = 0.0247
CAL_AY = -0.0249
CAL_AZ = 0.0278

# Queues for thread-safe data collection
right_queue = Queue()
left_queue = Queue()

print("=" * 70)
print(" DUAL GLOVE DATASET COLLECTOR (Time-Series)")
print("=" * 70)
print()
print("⚙️  CONFIGURATION:")
print(f"   - RIGHT glove: {RIGHT_GLOVE_PORT}")
print(f"   - LEFT glove: {LEFT_GLOVE_PORT}")
print(f"   - Sampling Rate: ~100 Hz per glove")
print(f"   - Duration per trial: {DURATION} seconds")
print(f"   - Expected samples per trial: ~300")
print(f"   - CSV Format: trial_id,sign_name,time_ms,flex1-10,ax_r,ay_r,az_r,gx_r,gy_r,gz_r,ax_l,ay_l,az_l,gx_l,gy_l,gz_l")
print()
print("🤚 SIGNS TO COLLECT:")
print("   - sample (open palm → closed fist)")
print("   - L, O, C, B, T (letter signs)")
print("   - good_morning, good_afternoon, good_evening (dynamic signs)")
print()
print("=" * 70)
print()

# Get sign name
sign_name = input("👉 Sign name (e.g., 'sample', 'L', 'O'): ").strip()
if not sign_name:
    print("❌ Sign name cannot be empty!")
    exit(1)

# Get starting trial number
start_trial = int(input(f"👉 Starting trial number (e.g., 1): ").strip())

# Get number of trials
num_trials = int(input(f"👉 How many '{sign_name}' trials to collect? (recommended: 30-50): ").strip())

print()
print(f"✅ Will collect {num_trials} trials of '{sign_name}' starting from trial {start_trial}")

# Create filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{sign_name}_dual_dataset_{timestamp}.csv"
print(f"📁 Output file: {filename}")
print()

# Connect to both ESP32s
input("⚠️  Press Enter to connect to both gloves...")

try:
    ser_right = serial.Serial(RIGHT_GLOVE_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ RIGHT glove (COM6) connected")
except Exception as e:
    print(f"❌ Error connecting to RIGHT glove: {e}")
    exit(1)

# Connect to LEFT glove (USB)
print(f"\n📡 Connecting to LEFT glove on {LEFT_GLOVE_PORT}...")

try:
    ser_left = serial.Serial(LEFT_GLOVE_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ LEFT glove (COM7) connected")
except Exception as e:
    print(f"❌ Error connecting to LEFT glove: {e}")
    ser_right.close()
    exit(1)

# Prepare CSV file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{sign_name}_dual_dataset_{timestamp}.csv"
csv_file = open(filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'trial_id', 'sign_name', 'time_ms',
    'flex_right_1', 'flex_right_2', 'flex_right_3', 'flex_right_4', 'flex_right_5',
    'flex_left_1', 'flex_left_2', 'flex_left_3', 'flex_left_4', 'flex_left_5',
    'ax_right', 'ay_right', 'az_right', 'gx_right', 'gy_right', 'gz_right',
    'ax_left', 'ay_left', 'az_left', 'gx_left', 'gy_left', 'gz_left'
])

def read_right_glove():
    """Thread to read RIGHT glove data"""
    while True:
        if ser_right.in_waiting > 0:
            try:
                line = ser_right.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    right_queue.put(line)
            except:
                pass

def read_left_glove():
    """Thread to read LEFT glove data"""
    while True:
        if ser_left.in_waiting > 0:
            try:
                line = ser_left.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    left_queue.put(line)
            except:
                pass

print("\n" + "=" * 80)
print(" READY TO COLLECT DUAL GLOVE DATA")
print("=" * 80)

# Start reader threads
right_thread = threading.Thread(target=read_right_glove, daemon=True)
left_thread = threading.Thread(target=read_left_glove, daemon=True)
right_thread.start()
left_thread.start()

for trial_num in range(start_trial, start_trial + num_trials):
    print(f"\n{'='*80}")
    print(f" TRIAL {trial_num}/{start_trial + num_trials - 1} - {sign_name.upper()} (BOTH HANDS)")
    print(f"{'='*80}")
    print(f"🤚 INSTRUCTION: Perform the '{sign_name}' sign with BOTH hands")
    print(f"💡 Duration: {DURATION} seconds")
    
    input(f"\n▶️  Press Enter to START recording trial {trial_num}...")
    
    # Clear queues
    while not right_queue.empty():
        right_queue.get()
    while not left_queue.empty():
        left_queue.get()
    
    # Start data collection on both gloves
    ser_right.write(b's')
    ser_left.write(b's')
    time.sleep(0.2)
    
    print(f"🔴 RECORDING for {DURATION} seconds...")
    print(f"   🤚 Perform the '{sign_name}' sign with BOTH HANDS!")
    
    trial_data = []
    start_time = time.time()
    first_timestamp = None
    
    while time.time() - start_time < DURATION:
        # Try to get synchronized data from both gloves
        right_data = None
        left_data = None
        
        if not right_queue.empty():
            right_line = right_queue.get()
            right_parts = right_line.split(',')
            if len(right_parts) == 12:
                right_data = right_parts
        
        if not left_queue.empty():
            left_line = left_queue.get()
            left_parts = left_line.split(',')
            if len(left_parts) == 12:
                left_data = left_parts
        
        # If we have data from both gloves, merge and save
        if right_data and left_data:
            timestamp_ms = float(right_data[0])
            
            if first_timestamp is None:
                first_timestamp = timestamp_ms
            
            relative_time = timestamp_ms - first_timestamp
            
            # Build merged row with calibration applied
            row = [
                trial_num,
                sign_name,
                relative_time,
                # RIGHT flex (5)
                int(right_data[1]), int(right_data[2]), int(right_data[3]),
                int(right_data[4]), int(right_data[5]),
                # LEFT flex (5)
                int(left_data[1]), int(left_data[2]), int(left_data[3]),
                int(left_data[4]), int(left_data[5]),
                # RIGHT MPU calibrated (6)
                float(right_data[6]) - CAL_AX, float(right_data[7]) - CAL_AY, float(right_data[8]) - CAL_AZ,
                float(right_data[9]) - CAL_GX, float(right_data[10]) - CAL_GY, float(right_data[11]) - CAL_GZ,
                # LEFT MPU calibrated (6)
                float(left_data[6]) - CAL_AX, float(left_data[7]) - CAL_AY, float(left_data[8]) - CAL_AZ,
                float(left_data[9]) - CAL_GX, float(left_data[10]) - CAL_GY, float(left_data[11]) - CAL_GZ
            ]
            
            csv_writer.writerow(row)
            trial_data.append(row)
        
        time.sleep(0.005)  # Small delay
    
    # Stop collection
    ser_right.write(b'x')
    ser_left.write(b'x')
    time.sleep(0.2)
    
    csv_file.flush()
    
    sample_count = len(trial_data)
    sampling_rate = sample_count / DURATION if DURATION > 0 else 0
    
    print(f"✅ Trial {trial_num} complete: {sample_count} samples collected")
    print(f"   Actual sampling rate: ~{sampling_rate:.1f} Hz")
    print("\n⏸️  Get ready for next trial...")
    time.sleep(1)

# Cleanup
csv_file.close()
ser_right.close()
ser_left.close()

print("\n" + "=" * 80)
print("✅ DATA COLLECTION COMPLETE!")
print("=" * 80)
print(f"\n📁 Saved to: {filename}")
print(f"📊 Total trials: {num_trials}")
print(f"💾 Ready for ML training!")
