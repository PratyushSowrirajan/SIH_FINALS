"""
MPU6050 Automatic Calibration Script
Places MPU flat on table and calculates gyro + accel offsets
"""

import serial
import time
import statistics

# Configuration
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
CALIBRATION_SAMPLES = 500  # ~5 seconds at 100Hz
EXPECTED_GRAVITY = 1.0  # Expected gravity on one axis when flat

def main():
    print("=" * 60)
    print("MPU6050 AUTOMATIC CALIBRATION")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Place MPU6050 FLAT on a STABLE surface!")
    print("   Do NOT move it during calibration!\n")
    
    input("Press Enter when MPU is stable and ready...")
    
    print(f"\n🔄 Connecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("✅ Connected!\n")
        
        # Clear any existing data
        ser.reset_input_buffer()
        
        # Send start command
        print("📡 Starting data collection...")
        ser.write(b's')
        time.sleep(0.5)
        
        # Collect samples
        ax_samples = []
        ay_samples = []
        az_samples = []
        gx_samples = []
        gy_samples = []
        gz_samples = []
        
        print(f"📊 Collecting {CALIBRATION_SAMPLES} samples (~5 seconds)...\n")
        
        sample_count = 0
        while sample_count < CALIBRATION_SAMPLES:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Check if it's data (starts with a number)
                    if line and line[0].isdigit():
                        parts = line.split(',')
                        if len(parts) == 7:  # Time,ax,ay,az,gx,gy,gz
                            ax_samples.append(float(parts[1]))
                            ay_samples.append(float(parts[2]))
                            az_samples.append(float(parts[3]))
                            gx_samples.append(float(parts[4]))
                            gy_samples.append(float(parts[5]))
                            gz_samples.append(float(parts[6]))
                            
                            sample_count += 1
                            if sample_count % 50 == 0:
                                print(f"  Collected {sample_count}/{CALIBRATION_SAMPLES} samples...")
                                
                except (UnicodeDecodeError, ValueError):
                    pass
        
        # Stop collection
        ser.write(b'x')
        ser.close()
        
        print("\n✅ Collection complete!\n")
        
        # Calculate offsets
        print("🧮 Calculating offsets...")
        print("-" * 60)
        
        # Gyroscope bias (should be ~0 when still)
        gx_bias = statistics.mean(gx_samples)
        gy_bias = statistics.mean(gy_samples)
        gz_bias = statistics.mean(gz_samples)
        
        # Accelerometer bias
        ax_bias = statistics.mean(ax_samples)
        ay_bias = statistics.mean(ay_samples)
        az_bias = statistics.mean(az_samples)
        
        # Determine which axis has gravity
        ax_abs = abs(ax_bias)
        ay_abs = abs(ay_bias)
        az_abs = abs(az_bias)
        
        # The axis with ~1g is the vertical one
        if az_abs > ax_abs and az_abs > ay_abs:
            # Z-axis is vertical (normal orientation)
            az_bias = az_bias - (EXPECTED_GRAVITY if az_bias > 0 else -EXPECTED_GRAVITY)
            gravity_axis = "Z"
        elif ay_abs > ax_abs and ay_abs > az_abs:
            # Y-axis is vertical
            ay_bias = ay_bias - (EXPECTED_GRAVITY if ay_bias > 0 else -EXPECTED_GRAVITY)
            gravity_axis = "Y"
        else:
            # X-axis is vertical
            ax_bias = ax_bias - (EXPECTED_GRAVITY if ax_bias > 0 else -EXPECTED_GRAVITY)
            gravity_axis = "X"
        
        print("\n📊 CALIBRATION RESULTS:")
        print("=" * 60)
        print(f"  Gravity detected on: {gravity_axis}-axis")
        print(f"\n  Gyroscope Offsets (deg/s):")
        print(f"    GX_OFFSET = {gx_bias:.3f}")
        print(f"    GY_OFFSET = {gy_bias:.3f}")
        print(f"    GZ_OFFSET = {gz_bias:.3f}")
        print(f"\n  Accelerometer Offsets (g):")
        print(f"    AX_OFFSET = {ax_bias:.4f}")
        print(f"    AY_OFFSET = {ay_bias:.4f}")
        print(f"    AZ_OFFSET = {az_bias:.4f}")
        print("=" * 60)
        
        # Save to file
        with open('calibration_offsets.txt', 'w') as f:
            f.write("// MPU6050 Calibration Offsets\n")
            f.write("// Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            f.write("// Gyroscope offsets (deg/s)\n")
            f.write(f"float GX_OFFSET = {gx_bias:.6f}f;\n")
            f.write(f"float GY_OFFSET = {gy_bias:.6f}f;\n")
            f.write(f"float GZ_OFFSET = {gz_bias:.6f}f;\n\n")
            f.write("// Accelerometer offsets (g)\n")
            f.write(f"float AX_OFFSET = {ax_bias:.6f}f;\n")
            f.write(f"float AY_OFFSET = {ay_bias:.6f}f;\n")
            f.write(f"float AZ_OFFSET = {az_bias:.6f}f;\n")
        
        print("\n💾 Offsets saved to: calibration_offsets.txt")
        print("\n✅ CALIBRATION COMPLETE!")
        print("\nNext step: These offsets will be applied to your ESP32 code.")
        
    except serial.SerialException as e:
        print(f"\n❌ Error: Could not open {SERIAL_PORT}")
        print(f"   {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Calibration interrupted by user")

if __name__ == "__main__":
    main()
