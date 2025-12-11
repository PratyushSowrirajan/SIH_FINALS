"""
Real-time HI gesture detector
Tests live gestures against trained model
"""

import serial
import time
import numpy as np
import pickle

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
RECORDING_DURATION = 2.0  # Match training duration

print("=" * 70)
print(" REAL-TIME HI GESTURE DETECTOR")
print("=" * 70)

# Load trained model
print("\n📂 Loading trained model...")
try:
    with open('hi_gesture_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    template = model['template']
    threshold = model['threshold']
    target_length = model['target_length']
    
    print(f"✅ Model loaded successfully")
    print(f"   Trained on: {model['trained_on']}")
    print(f"   Training accuracy: {model['accuracy']:.1f}%")
    print(f"   Detection threshold: {threshold:.2f}")
except FileNotFoundError:
    print("❌ Model not found! Run 'train_hi_model.py' first.")
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
    print("\n🎯 Get ready...")
    time.sleep(1)
    print("🔴 RECORDING... Do your gesture NOW!")
    
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
                    if len(parts) == 7:
                        ax = float(parts[1])
                        ay = float(parts[2])
                        az = float(parts[3])
                        gx = float(parts[4])
                        gy = float(parts[5])
                        gz = float(parts[6])
                        data.append([ax, ay, az, gx, gy, gz])
            except (UnicodeDecodeError, ValueError, IndexError):
                pass
    
    # Stop recording
    ser.write(b'x')
    time.sleep(0.2)
    
    if len(data) < 50:
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
    is_hi = distance < threshold
    confidence = abs(threshold - distance) / threshold * 100
    
    print(f"\n📊 RESULTS:")
    print(f"   Samples collected: {len(data)}")
    print(f"   Distance to template: {distance:.2f}")
    print(f"   Threshold: {threshold:.2f}")
    
    if is_hi:
        print(f"\n✅ DETECTED: HI GESTURE! 👋")
        print(f"   Confidence: {min(confidence, 100):.1f}%")
    else:
        print(f"\n❌ NOT a HI gesture")
        print(f"   Distance too far from template")
    
    return is_hi, distance, confidence

print("\n" + "=" * 70)
print(" READY TO DETECT GESTURES")
print("=" * 70)
print("💡 Do the HI gesture: CENTER → LEFT → CENTER")
print("   Or do other gestures to test false positives")
print("\n   Press Ctrl+C to quit")
print("=" * 70)

test_number = 1
try:
    while True:
        input(f"\n▶️  Test #{test_number}: Press Enter to record a gesture...")
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
