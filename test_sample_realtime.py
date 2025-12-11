"""
Real-time 'sample' sign detector (open palm -> closed fist)
Tests live gestures against trained model
"""

import serial
import time
import numpy as np
import pickle

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
RECORDING_DURATION = 3.0  # Match training duration

print("=" * 70)
print(" REAL-TIME 'SAMPLE' SIGN DETECTOR")
print("=" * 70)

# Load trained model
print("\n📂 Loading trained model...")
try:
    with open('sample_sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    template = model['template']
    threshold = model['threshold']
    target_length = model['target_length']
    sign_name = model['sign_name']
    
    print(f"✅ Model loaded successfully")
    print(f"   Sign: '{sign_name}'")
    print(f"   Trained on: {model['trained_on']}")
    print(f"   Training accuracy: {model['accuracy']:.1f}%")
    print(f"   Detection threshold: {threshold:.2f}")
    print(f"   Trained on {model['num_trials']} trials")
except FileNotFoundError:
    print("❌ Model not found! Run 'train_sample_model.py' first.")
    exit(1)

# Connect to ESP32
print("\n🔌 Connecting to ESP32...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Connected to ESP32 on COM5")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

def euclidean_distance(a, b):
    """Calculate Euclidean distance"""
    if len(a) != len(b):
        indices = np.linspace(0, len(b) - 1, len(a))
        b = np.array([b[int(i)] for i in indices])
    return np.sqrt(np.sum((a - b) ** 2))

def record_and_classify():
    """Record one gesture and classify it"""
    print("\n🤚 Get ready to perform the sign...")
    time.sleep(1)
    print("🔴 RECORDING... Do the SAMPLE sign NOW!")
    print("   (Open palm → Close fist → Open palm)")
    
    # Start recording
    ser.write(b's')
    time.sleep(0.2)
    
    data = []
    start_time = time.time()
    
    while time.time() - start_time < RECORDING_DURATION:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    parts = line.split(',')
                    if len(parts) == 12:  # Time,flex1-5,ax,ay,az,gx,gy,gz
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
                        data.append([flex1, flex2, flex3, flex4, flex5, ax, ay, az, gx, gy, gz])
            except (UnicodeDecodeError, ValueError, IndexError):
                pass
    
    # Stop recording
    ser.write(b'x')
    time.sleep(0.2)
    
    if len(data) < 100:
        print("⚠️  Not enough data collected, try again")
        return None
    
    # Convert to numpy array
    gesture_data = np.array(data)
    
    # Resample to target length
    indices = np.linspace(0, len(gesture_data) - 1, target_length)
    resampled = np.array([gesture_data[int(i)] for i in indices])
    
    # Calculate distance to template
    distance = euclidean_distance(template, resampled)
    
    # Classify
    is_sample = distance < threshold
    confidence = max(0, min(100, (1 - distance / threshold) * 100))
    
    print(f"\n📊 RESULTS:")
    print(f"   Samples collected: {len(data)}")
    print(f"   Distance to template: {distance:.2f}")
    print(f"   Threshold: {threshold:.2f}")
    
    if is_sample:
        print(f"\n✅ DETECTED: '{sign_name.upper()}' SIGN! 🤚✊")
        print(f"   Confidence: {confidence:.1f}%")
    else:
        print(f"\n❌ NOT a '{sign_name}' sign")
        print(f"   Distance too far from template")
        print(f"   Similarity: {max(0, 100 - (distance - threshold) / threshold * 100):.1f}%")
    
    return is_sample, distance, confidence

print("\n" + "=" * 70)
print(" READY TO DETECT 'SAMPLE' SIGN")
print("=" * 70)
print("💡 Do the sign: Open palm → Close fist → Open palm (repeat)")
print("   Duration: 3 seconds per test")
print("\n   Press Ctrl+C to quit")
print("=" * 70)

test_number = 1
try:
    while True:
        input(f"\n▶️  Test #{test_number}: Press Enter to record...")
        result = record_and_classify()
        if result:
            test_number += 1
        print("\n" + "-" * 70)

except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print(" TESTING COMPLETE")
    print("=" * 70)
    print("✅ Detector stopped")
finally:
    ser.close()
