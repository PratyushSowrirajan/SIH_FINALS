"""
Center-Right-Center Movement Comparison Test
Collects two repetitions and compares their similarity
"""

import serial
import time
import csv
from datetime import datetime

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200

print("=" * 70)
print(" CENTER → RIGHT → CENTER Movement Test")
print("=" * 70)
print("\n🎯 TEST PROCEDURE:")
print("   1. Hold MPU steady at CENTER position")
print("   2. Press Enter to start recording")
print("   3. Move to RIGHT, then back to CENTER (take ~2-3 seconds)")
print("   4. Stop recording")
print("   5. REPEAT the same movement")
print("   6. Script will compare both movements")
print("\n⚠️  Make sure ESP32 is connected to COM5\n")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Connected to ESP32\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

def collect_movement(movement_num):
    """Collect one C-R-C movement"""
    input(f"\n▶️  MOVEMENT {movement_num}: Position MPU at CENTER, then press Enter...")
    
    print(f"🔴 Recording movement {movement_num}... (Press Ctrl+C when done)")
    
    ser.write(b's')  # Start collection
    time.sleep(0.3)
    
    data = []
    start_time = None
    
    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line and line[0].isdigit():
                        parts = line.split(',')
                        if len(parts) == 7:
                            if start_time is None:
                                start_time = float(parts[0])
                            
                            rel_time = float(parts[0]) - start_time
                            data.append({
                                'time': rel_time,
                                'ax': float(parts[1]),
                                'ay': float(parts[2]),
                                'az': float(parts[3]),
                                'gx': float(parts[4]),
                                'gy': float(parts[5]),
                                'gz': float(parts[6])
                            })
                except (UnicodeDecodeError, ValueError, IndexError):
                    pass
    except KeyboardInterrupt:
        ser.write(b'x')  # Stop collection
        print(f"\n✅ Movement {movement_num} recorded: {len(data)} samples")
        return data

# Collect two movements
print("\n" + "=" * 70)
print(" COLLECTING MOVEMENT 1")
print("=" * 70)
movement1 = collect_movement(1)

print("\n" + "=" * 70)
print(" COLLECTING MOVEMENT 2")
print("=" * 70)
print("   (Try to repeat the EXACT same movement)")
movement2 = collect_movement(2)

# Save data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename1 = f"movement1_CRC_{timestamp}.csv"
filename2 = f"movement2_CRC_{timestamp}.csv"

with open(filename1, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    writer.writeheader()
    writer.writerows(movement1)

with open(filename2, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    writer.writeheader()
    writer.writerows(movement2)

print(f"\n💾 Saved: {filename1}")
print(f"💾 Saved: {filename2}")

# Quick comparison
print("\n" + "=" * 70)
print(" COMPARISON ANALYSIS")
print("=" * 70)

def analyze_movement(data, name):
    """Calculate statistics for a movement"""
    ax = [d['ax'] for d in data]
    ay = [d['ay'] for d in data]
    az = [d['az'] for d in data]
    gx = [d['gx'] for d in data]
    gy = [d['gy'] for d in data]
    gz = [d['gz'] for d in data]
    
    print(f"\n📊 {name}:")
    print(f"   Duration: {data[-1]['time']:.2f} ms")
    print(f"   Samples: {len(data)}")
    print(f"   Accel ranges: X=[{min(ax):.3f}, {max(ax):.3f}]  "
          f"Y=[{min(ay):.3f}, {max(ay):.3f}]  "
          f"Z=[{min(az):.3f}, {max(az):.3f}]")
    print(f"   Gyro ranges:  X=[{min(gx):.1f}, {max(gx):.1f}]  "
          f"Y=[{min(gy):.1f}, {max(gy):.1f}]  "
          f"Z=[{min(gz):.1f}, {max(gz):.1f}]")
    
    return {
        'ax_range': max(ax) - min(ax),
        'ay_range': max(ay) - min(ay),
        'az_range': max(az) - min(az),
        'gx_range': max(gx) - min(gx),
        'gy_range': max(gy) - min(gy),
        'gz_range': max(gz) - min(gz),
    }

stats1 = analyze_movement(movement1, "MOVEMENT 1")
stats2 = analyze_movement(movement2, "MOVEMENT 2")

print("\n" + "=" * 70)
print(" REPEATABILITY CHECK")
print("=" * 70)
print("\n🔍 Range differences (lower = more consistent):")
for key in stats1.keys():
    diff = abs(stats1[key] - stats2[key])
    axis_name = key.replace('_range', '').upper()
    pct_diff = (diff / max(stats1[key], stats2[key]) * 100) if max(stats1[key], stats2[key]) > 0 else 0
    print(f"   {axis_name}: {diff:.3f} ({pct_diff:.1f}% difference)")

print("\n💡 INTERPRETATION:")
print("   - If differences are <20%, movements are VERY similar")
print("   - If 20-40%, movements are reasonably consistent")
print("   - If >40%, movements vary significantly")
print("\n✅ Test complete!")

ser.close()
