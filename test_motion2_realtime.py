"""
Real-time 'motion2' gesture detector
Connects to ESP32, captures sensor data, and detects the gesture
"""

import serial
import time
import pickle
import numpy as np
import pandas as pd

print("=" * 70)
print(" REAL-TIME 'MOTION2' GESTURE DETECTOR")
print("=" * 70)

# Load trained model
print("\n📂 Loading trained model...")
with open('motion2_gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")
print(f"   Gesture: '{model['sign_name']}'")
print(f"   Trained on: {model['trained_on']}")
print(f"   Training accuracy: {model['accuracy']:.1f}%")
print(f"   Detection threshold: {model['threshold']:.2f}")
print(f"   Trained on {model['num_trials']} trials")

# Connect to ESP32
print("\n🔌 Connecting to ESP32...")
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)
print("✅ Connected to ESP32 on COM6")

def read_sensor_data(duration_sec=3):
    """Read sensor data for specified duration"""
    samples = []
    start_time = time.time()
    
    # Clear buffer
    ser.reset_input_buffer()
    
    while time.time() - start_time < duration_sec:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line or 'flex1' in line or 'Calibration' in line:
                continue
                
            parts = line.split(',')
            if len(parts) >= 11:
                flex_values = [int(x) for x in parts[:5]]
                ax, ay, az = float(parts[5]), float(parts[6]), float(parts[7])
                gx, gy, gz = float(parts[8]), float(parts[9]), float(parts[10])
                
                samples.append({
                    'flex1': flex_values[0],
                    'flex2': flex_values[1],
                    'flex3': flex_values[2],
                    'flex4': flex_values[3],
                    'flex5': flex_values[4],
                    'ax': ax, 'ay': ay, 'az': az,
                    'gx': gx, 'gy': gy, 'gz': gz
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(samples)

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two sequences"""
    if len(a) != len(b):
        indices = np.linspace(0, len(b) - 1, len(a))
        b = np.array([b[int(i)] for i in indices])
    return np.sqrt(np.sum((a - b) ** 2))

def detect_gesture(data, model):
    """Detect if gesture matches the trained model"""
    if len(data) < 10:
        return False, float('inf')
    
    # Extract features in same format as training
    features = np.column_stack([
        data['flex1'].values,
        data['flex2'].values,
        data['flex3'].values,
        data['flex4'].values,
        data['flex5'].values,
        data['ax'].values,
        data['ay'].values,
        data['az'].values,
        data['gx'].values,
        data['gy'].values,
        data['gz'].values
    ])
    
    # Resample to target length
    TARGET_LENGTH = model['target_length']
    indices = np.linspace(0, len(features) - 1, TARGET_LENGTH)
    resampled = np.array([features[int(i)] for i in indices])
    
    # Calculate distance to template
    distance = euclidean_distance(model['template'], resampled)
    
    # Check if below threshold
    detected = distance < model['threshold']
    
    return detected, distance

print("\n" + "=" * 70)
print(" READY TO TEST")
print("=" * 70)
print("\nInstructions:")
print("  - Press Enter to record and test a gesture")
print("  - Press 'q' and Enter to quit")
print("=" * 70)

try:
    while True:
        user_input = input("\n▶️  Press Enter to test (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            print("\n👋 Goodbye!")
            break
        
        print("\n🤚 Get ready to perform the gesture...")
        time.sleep(1)
        
        print("🔴 RECORDING... Do the MOTION2 gesture NOW!")
        data = read_sensor_data(duration_sec=3)
        
        print(f"✅ Recording complete ({len(data)} samples)")
        
        # Detect gesture
        detected, distance = detect_gesture(data, model)
        
        print(f"\n📊 Analysis:")
        print(f"   Distance to template: {distance:.2f}")
        print(f"   Detection threshold: {model['threshold']:.2f}")
        
        if detected:
            print(f"\n✅ ✅ ✅  'MOTION2' DETECTED!  ✅ ✅ ✅")
            print(f"   Confidence: {max(0, 100 * (1 - distance / model['threshold'])):.1f}%")
        else:
            print(f"\n❌ Not detected")
            print(f"   Distance exceeds threshold by {distance - model['threshold']:.2f}")
            
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
finally:
    ser.close()
    print("🔌 Serial connection closed")
