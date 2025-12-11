"""
Real-time Sign Language Recognition Web Interface
Elegant, simple interface to display recognized signs as text
"""

from flask import Flask, render_template
from flask_socketio import SocketIO
import serial
import numpy as np
import pickle
import threading
import time
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign-language-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
RECORDING_DURATION = 3.0

# Global state
is_recognizing = False
ser = None
loaded_models = {}

def load_all_models():
    """Load all available gesture models"""
    models = {}
    model_files = [f for f in os.listdir('.') if f.endswith('_model.pkl') or f.endswith('_gesture_model.pkl')]
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                sign_name = model['sign_name']
                models[sign_name] = model
                print(f"✅ Loaded model: {sign_name}")
        except Exception as e:
            print(f"⚠️  Error loading {model_file}: {e}")
    
    return models

def euclidean_distance(a, b):
    """Calculate Euclidean distance"""
    if len(a) != len(b):
        indices = np.linspace(0, len(b) - 1, len(a))
        b = np.array([b[int(i)] for i in indices])
    return np.sqrt(np.sum((a - b) ** 2))

def classify_gesture(gesture_data, models):
    """Classify gesture against all loaded models"""
    if len(gesture_data) < 50:
        return None, 0, "Insufficient data"
    
    results = []
    
    for sign_name, model in models.items():
        template = model['template']
        threshold = model['threshold']
        target_length = model['target_length']
        
        # Resample
        indices = np.linspace(0, len(gesture_data) - 1, target_length)
        resampled = np.array([gesture_data[int(i)] for i in indices])
        
        # Calculate distance
        distance = euclidean_distance(template, resampled)
        
        # Calculate confidence
        confidence = max(0, min(100, 100 * (1 - distance / threshold)))
        
        results.append({
            'sign': sign_name,
            'distance': distance,
            'threshold': threshold,
            'confidence': confidence,
            'match': distance < threshold
        })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Return best match if any
    if results and results[0]['match']:
        return results[0]['sign'], results[0]['confidence'], "Recognized"
    else:
        return None, 0, "No match found"

def recognition_loop():
    """Background thread for continuous recognition"""
    global is_recognizing, ser
    
    while True:
        if is_recognizing and ser and ser.is_open:
            try:
                # Start recording
                ser.write(b's')
                time.sleep(0.2)
                
                data = []
                start_time = time.time()
                
                socketio.emit('status', {'message': 'Recording...', 'recording': True})
                
                # Collect data
                while time.time() - start_time < RECORDING_DURATION:
                    if ser.in_waiting > 0:
                        try:
                            line = ser.readline().decode('utf-8').strip()
                            if line and line[0].isdigit():
                                parts = line.split(',')
                                if len(parts) == 12:
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
                
                socketio.emit('status', {'message': 'Processing...', 'recording': False})
                
                # Classify
                if len(data) >= 50:
                    gesture_data = np.array(data)
                    sign, confidence, status = classify_gesture(gesture_data, loaded_models)
                    
                    if sign:
                        socketio.emit('sign_detected', {
                            'sign': sign.upper(),
                            'confidence': round(confidence, 1),
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'samples': len(data)
                        })
                    else:
                        socketio.emit('no_match', {
                            'message': status,
                            'samples': len(data)
                        })
                else:
                    socketio.emit('error', {'message': 'Not enough data collected'})
                
                time.sleep(0.5)  # Small delay between recognitions
                
            except Exception as e:
                socketio.emit('error', {'message': str(e)})
                time.sleep(1)
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    model_names = list(loaded_models.keys())
    socketio.emit('models_loaded', {'models': model_names, 'count': len(model_names)})

@socketio.on('start_recognition')
def handle_start():
    """Start continuous recognition"""
    global is_recognizing, ser
    
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
        
        is_recognizing = True
        socketio.emit('recognition_started', {'message': 'Recognition started'})
    except Exception as e:
        socketio.emit('error', {'message': f'Failed to connect: {str(e)}'})

@socketio.on('stop_recognition')
def handle_stop():
    """Stop recognition"""
    global is_recognizing
    is_recognizing = False
    socketio.emit('recognition_stopped', {'message': 'Recognition stopped'})

if __name__ == '__main__':
    print("=" * 70)
    print(" SIGN LANGUAGE RECOGNITION WEB INTERFACE")
    print("=" * 70)
    
    # Load models
    print("\n📂 Loading gesture models...")
    loaded_models = load_all_models()
    
    if not loaded_models:
        print("⚠️  No models found! Train some models first.")
        exit(1)
    
    print(f"\n✅ Loaded {len(loaded_models)} models: {', '.join(loaded_models.keys())}")
    
    # Start recognition thread
    recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
    recognition_thread.start()
    
    print("\n🌐 Starting web server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("=" * 70)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
