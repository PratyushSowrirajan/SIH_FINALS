"""
Collect calibration verification data
Automatically saves to after_cal.csv
"""

import serial
import csv
import time
from datetime import datetime

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
OUTPUT_FILE = 'after_cal.csv'
SAMPLE_DURATION = 5  # seconds

print("=" * 60)
print("CALIBRATION VERIFICATION - Collecting still sensor data")
print("=" * 60)
print(f"\nWill collect data for {SAMPLE_DURATION} seconds...")
print("MPU should be FLAT and STILL on table!\n")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Connected to ESP32\n")
    
    # Clear buffer
    ser.reset_input_buffer()
    
    # Send start command
    print("📡 Starting data collection...")
    ser.write(b's')
    time.sleep(0.5)
    
    # Open CSV file
    csv_file = open(OUTPUT_FILE, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    header_written = False
    
    start_time = time.time()
    sample_count = 0
    
    print(f"⏱️  Collecting for {SAMPLE_DURATION} seconds...\n")
    
    while (time.time() - start_time) < SAMPLE_DURATION:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    # Write header
                    if 'Time(ms)' in line and not header_written:
                        csv_writer.writerow(line.split(','))
                        header_written = True
                        print("📊 Header: " + line)
                    # Write data
                    elif line and line[0].isdigit():
                        csv_writer.writerow(line.split(','))
                        csv_file.flush()
                        sample_count += 1
                        if sample_count % 50 == 0:
                            print(f"  Collected {sample_count} samples...")
            except UnicodeDecodeError:
                pass
    
    # Stop collection
    ser.write(b'x')
    csv_file.close()
    ser.close()
    
    print(f"\n✅ COMPLETE! Collected {sample_count} samples")
    print(f"💾 Data saved to: {OUTPUT_FILE}")
    print("\n📊 You can now verify the calibration quality!")
    print("   - Gyro values should be near 0 when still")
    print("   - Accel should show ~1g on one axis, ~0 on others\n")
    
except serial.SerialException as e:
    print(f"❌ Error: Could not open {SERIAL_PORT}")
    print(f"   {e}")
except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    if csv_file:
        csv_file.close()
