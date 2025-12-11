"""
Real-time MPU6050 Axis Sanity Check
Shows live sensor values to verify axis orientation
"""

import serial
import time
from collections import deque

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200

print("=" * 70)
print(" MPU6050 AXIS SANITY CHECK - Real-Time Monitoring")
print("=" * 70)
print("\n🎯 INSTRUCTIONS:")
print("   1. Watch the live values below")
print("   2. Move your hand/MPU in different directions:")
print("      - Up/Down")
print("      - Left/Right") 
print("      - Forward/Backward")
print("      - Rotate clockwise/counterclockwise")
print("   3. Note which axes respond to which movements")
print("\n⚠️  Press Ctrl+C to stop\n")

input("Press Enter to start monitoring...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    
    # Send start command
    ser.write(b's')
    time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("📡 LIVE DATA (press Ctrl+C to stop)")
    print("=" * 70)
    print()
    
    # Keep recent values for min/max tracking
    ax_history = deque(maxlen=50)
    ay_history = deque(maxlen=50)
    az_history = deque(maxlen=50)
    gx_history = deque(maxlen=50)
    gy_history = deque(maxlen=50)
    gz_history = deque(maxlen=50)
    
    line_count = 0
    
    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                
                # Parse data line
                if line and line[0].isdigit():
                    parts = line.split(',')
                    if len(parts) == 7:
                        timestamp = parts[0]
                        ax = float(parts[1])
                        ay = float(parts[2])
                        az = float(parts[3])
                        gx = float(parts[4])
                        gy = float(parts[5])
                        gz = float(parts[6])
                        
                        # Update history
                        ax_history.append(ax)
                        ay_history.append(ay)
                        az_history.append(az)
                        gx_history.append(gx)
                        gy_history.append(gy)
                        gz_history.append(gz)
                        
                        # Print every 5th reading (reduce clutter)
                        if line_count % 5 == 0:
                            print(f"\r{'  ' * 80}", end='')  # Clear line
                            print(f"\r📊 ACCEL: X={ax:+.3f}g  Y={ay:+.3f}g  Z={az:+.3f}g  |  "
                                  f"GYRO: X={gx:+6.1f}°/s  Y={gy:+6.1f}°/s  Z={gz:+6.1f}°/s", 
                                  end='', flush=True)
                        
                        line_count += 1
                        
                        # Every 50 samples, show ranges
                        if line_count % 50 == 0 and len(ax_history) > 10:
                            print("\n")
                            print("  📈 Recent ranges (last ~0.5 sec):")
                            print(f"     ACCEL X: {min(ax_history):+.3f} to {max(ax_history):+.3f}g  "
                                  f"(Δ={max(ax_history)-min(ax_history):.3f})")
                            print(f"     ACCEL Y: {min(ay_history):+.3f} to {max(ay_history):+.3f}g  "
                                  f"(Δ={max(ay_history)-min(ay_history):.3f})")
                            print(f"     ACCEL Z: {min(az_history):+.3f} to {max(az_history):+.3f}g  "
                                  f"(Δ={max(az_history)-min(az_history):.3f})")
                            print(f"     GYRO  X: {min(gx_history):+6.1f} to {max(gx_history):+6.1f}°/s  "
                                  f"(Δ={max(gx_history)-min(gx_history):.1f})")
                            print(f"     GYRO  Y: {min(gy_history):+6.1f} to {max(gy_history):+6.1f}°/s  "
                                  f"(Δ={max(gy_history)-min(gy_history):.1f})")
                            print(f"     GYRO  Z: {min(gz_history):+6.1f} to {max(gz_history):+6.1f}°/s  "
                                  f"(Δ={max(gz_history)-min(gz_history):.1f})")
                            print()
                            
            except (UnicodeDecodeError, ValueError, IndexError):
                pass
        
        time.sleep(0.01)
        
except serial.SerialException as e:
    print(f"\n❌ Error: {e}")
except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("✅ Monitoring stopped")
    print("=" * 70)
    print("\n📝 WHAT TO LOOK FOR:")
    print("   - When you move UP/DOWN, which accel axis changes most?")
    print("   - When you move LEFT/RIGHT, which accel axis changes?")
    print("   - When you rotate, which gyro axis shows highest values?")
    print("   - Do the same gestures give similar patterns? (repeatability)")
    print("\n💡 TIP: The axis with ~1g when still is pointing DOWN (gravity)")
    print("=" * 70)
finally:
    if 'ser' in locals() and ser.is_open:
        ser.write(b'x')
        ser.close()
