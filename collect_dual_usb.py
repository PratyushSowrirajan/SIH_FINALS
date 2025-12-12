"""
Dual Glove Dataset Collector - BOTH via USB
RIGHT glove: COM7
LEFT glove: COM6
Merges data into single CSV with 22 columns
"""

import serial
import time
import csv
import threading
from datetime import datetime
from queue import Queue

# Configuration
RIGHT_PORT = 'COM7'
LEFT_PORT = 'COM6'
BAUD_RATE = 115200
DURATION_SECONDS = 3.0

print("=" * 80)
print(" DUAL GLOVE USB DATASET COLLECTOR")
print("=" * 80)
print(f"\n⚙️  CONFIGURATION:")
print(f"   - RIGHT glove: USB on {RIGHT_PORT}")
print(f"   - LEFT glove: USB on {LEFT_PORT}")
print(f"   - Duration per trial: {DURATION_SECONDS} seconds")
print(f"   - CSV: 22 columns (10 flex + 12 MPU)")

# Get parameters
sign_name = input("\n👉 Sign name: ").strip()
start_trial = int(input(f"👉 Starting trial number: ").strip())
num_trials = int(input(f"👉 How many trials? (recommended: 40): ").strip())

print(f"\n✅ Will collect {num_trials} trials of '{sign_name}' from BOTH gloves")

# Connect
print("\n🔌 Connecting to gloves...")
try:
    ser_right = serial.Serial(RIGHT_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)
    print("✅ RIGHT glove connected")
except Exception as e:
    print(f"❌ RIGHT glove error: {e}")
    exit(1)

try:
    ser_left = serial.Serial(LEFT_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)
    print("✅ LEFT glove connected")
except Exception as e:
    print(f"❌ LEFT glove error: {e}")
    ser_right.close()
    exit(1)

# CSV
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

right_queue = Queue()
left_queue = Queue()

def read_right():
    while True:
        if ser_right.in_waiting > 0:
            try:
                line = ser_right.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    right_queue.put(line)
            except:
                pass

def read_left():
    while True:
        if ser_left.in_waiting > 0:
            try:
                line = ser_left.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    left_queue.put(line)
            except:
                pass

print("\n" + "=" * 80)
print(" READY TO COLLECT")
print("=" * 80)

for trial_num in range(start_trial, start_trial + num_trials):
    print(f"\n{'='*80}")
    print(f" TRIAL {trial_num}/{start_trial + num_trials - 1} - {sign_name.upper()}")
    print(f"{'='*80}")
    
    input(f"\n▶️  Press Enter to START trial {trial_num}...")
    
    # Clear queues
    while not right_queue.empty(): right_queue.get()
    while not left_queue.empty(): left_queue.get()
    
    # Start collection
    ser_right.write(b's')
    ser_left.write(b's')
    time.sleep(0.2)
    
    print(f"🔴 RECORDING for {DURATION_SECONDS} seconds...")
    
    # Start threads
    right_thread = threading.Thread(target=read_right, daemon=True)
    left_thread = threading.Thread(target=read_left, daemon=True)
    right_thread.start()
    left_thread.start()
    
    trial_data = []
    start_time = time.time()
    first_timestamp = None
    
    while time.time() - start_time < DURATION_SECONDS:
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
        
        if right_data and left_data:
            timestamp_ms = float(right_data[0])
            
            if first_timestamp is None:
                first_timestamp = timestamp_ms
            
            relative_time = timestamp_ms - first_timestamp
            
            row = [
                trial_num, sign_name, relative_time,
                # RIGHT flex
                int(right_data[1]), int(right_data[2]), int(right_data[3]),
                int(right_data[4]), int(right_data[5]),
                # LEFT flex
                int(left_data[1]), int(left_data[2]), int(left_data[3]),
                int(left_data[4]), int(left_data[5]),
                # RIGHT MPU
                float(right_data[6]), float(right_data[7]), float(right_data[8]),
                float(right_data[9]), float(right_data[10]), float(right_data[11]),
                # LEFT MPU
                float(left_data[6]), float(left_data[7]), float(left_data[8]),
                float(left_data[9]), float(left_data[10]), float(left_data[11])
            ]
            
            csv_writer.writerow(row)
            trial_data.append(row)
        
        time.sleep(0.005)
    
    # Stop
    ser_right.write(b'x')
    ser_left.write(b'x')
    time.sleep(0.2)
    csv_file.flush()
    
    print(f"✅ Trial {trial_num} complete: {len(trial_data)} samples")
    time.sleep(1)

csv_file.close()
ser_right.close()
ser_left.close()

print("\n" + "=" * 80)
print("✅ COMPLETE!")
print("=" * 80)
print(f"\n📁 Saved: {filename}")
