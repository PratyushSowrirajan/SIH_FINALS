"""
MPU6050 Real-Time Plotter with Matplotlib
Visual axis sanity check with live graphs
"""

import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
MAX_POINTS = 200  # Show last 2 seconds of data

# Data storage
time_data = deque(maxlen=MAX_POINTS)
ax_data = deque(maxlen=MAX_POINTS)
ay_data = deque(maxlen=MAX_POINTS)
az_data = deque(maxlen=MAX_POINTS)
gx_data = deque(maxlen=MAX_POINTS)
gy_data = deque(maxlen=MAX_POINTS)
gz_data = deque(maxlen=MAX_POINTS)

start_time = None
ser = None

print("=" * 70)
print(" MPU6050 REAL-TIME VISUALIZER")
print("=" * 70)
print("\n🎯 INSTRUCTIONS:")
print("   1. Two graphs will open: ACCELEROMETER (top) and GYROSCOPE (bottom)")
print("   2. Move your hand/MPU and watch the lines respond")
print("   3. Test repeatability: Do the SAME gesture twice")
print("   4. Close the graph window to stop")
print("\n⚠️  Starting in 3 seconds...\n")

# Initialize serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.write(b's')  # Start data collection
    time.sleep(0.5)
    print("✅ Connected to ESP32 on COM5\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Setup plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('MPU6050 Real-Time Axis Sanity Check', fontsize=16, fontweight='bold')

# Accelerometer plot
line_ax, = ax1.plot([], [], 'r-', label='Accel X', linewidth=2)
line_ay, = ax1.plot([], [], 'g-', label='Accel Y', linewidth=2)
line_az, = ax1.plot([], [], 'b-', label='Accel Z', linewidth=2)
ax1.set_ylabel('Acceleration (g)', fontsize=12)
ax1.set_ylim(-2, 2)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Gyroscope plot
line_gx, = ax2.plot([], [], 'r-', label='Gyro X', linewidth=2)
line_gy, = ax2.plot([], [], 'g-', label='Gyro Y', linewidth=2)
line_gz, = ax2.plot([], [], 'b-', label='Gyro Z', linewidth=2)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Angular Velocity (°/s)', fontsize=12)
ax2.set_ylim(-300, 300)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

def init():
    line_ax.set_data([], [])
    line_ay.set_data([], [])
    line_az.set_data([], [])
    line_gx.set_data([], [])
    line_gy.set_data([], [])
    line_gz.set_data([], [])
    return line_ax, line_ay, line_az, line_gx, line_gy, line_gz

def update(frame):
    global start_time
    
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            
            if line and line[0].isdigit():
                parts = line.split(',')
                if len(parts) == 7:
                    timestamp_ms = float(parts[0])
                    
                    if start_time is None:
                        start_time = timestamp_ms
                    
                    rel_time = (timestamp_ms - start_time) / 1000.0  # Convert to seconds
                    
                    time_data.append(rel_time)
                    ax_data.append(float(parts[1]))
                    ay_data.append(float(parts[2]))
                    az_data.append(float(parts[3]))
                    gx_data.append(float(parts[4]))
                    gy_data.append(float(parts[5]))
                    gz_data.append(float(parts[6]))
                    
        except (UnicodeDecodeError, ValueError, IndexError):
            pass
    
    if len(time_data) > 0:
        # Update accelerometer lines
        line_ax.set_data(list(time_data), list(ax_data))
        line_ay.set_data(list(time_data), list(ay_data))
        line_az.set_data(list(time_data), list(az_data))
        
        # Update gyroscope lines
        line_gx.set_data(list(time_data), list(gx_data))
        line_gy.set_data(list(time_data), list(gy_data))
        line_gz.set_data(list(time_data), list(gz_data))
        
        # Auto-adjust x-axis
        if len(time_data) > 1:
            ax1.set_xlim(time_data[0], time_data[-1])
            ax2.set_xlim(time_data[0], time_data[-1])
    
    return line_ax, line_ay, line_az, line_gx, line_gy, line_gz

ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=20, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    if ser and ser.is_open:
        ser.write(b'x')
        ser.close()
    print("\n✅ Visualization stopped")
