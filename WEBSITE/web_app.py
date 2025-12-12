"""
ISL (Indian Sign Language) Recognition Web Application
=======================================================
A Flask web app with:
- Real-time ISL alphabet to text conversion
- Tutorial videos for learning ISL
- Alphabet and Sentence recognition modes

Reference Video: https://www.youtube.com/watch?v=qcdivQfA41Y
Train model with ISL data using: python collect_isl_complete.py
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque
import time

app = Flask(__name__)

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
def load_models():
    """Load trained models"""
    try:
        alphabet_model = joblib.load("alphabet_model.pkl")
        alphabet_labels = joblib.load("alphabet_labels.pkl")
        sentence_model = joblib.load("sentence_model.pkl")
        sentence_labels = joblib.load("sentence_labels.pkl")
        return alphabet_model, alphabet_labels, sentence_model, sentence_labels, None
    except Exception as e:
        return None, None, None, None, str(e)

alphabet_model, alphabet_labels, sentence_model, sentence_labels, error = load_models()

# ---------------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lowered for better detection
    min_tracking_confidence=0.5     # Lowered for better tracking
)
mp_draw = mp.solutions.drawing_utils

# ---------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------
current_mode = 'alphabet'  # 'alphabet' or 'sentence'
detected_text = ""
word_builder = []
last_alphabet = ""
last_time = time.time()
last_stable_time = time.time()
alphabet_history = deque(maxlen=15)  # Balanced for responsiveness
sentence_history = deque(maxlen=15)
min_confidence = 0.50  # Lower threshold for better detection

# Stability tracking for automatic letter addition
stable_alphabet = ""
stable_count = 0
required_stability = 8  # Reduced for faster detection

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_landmarks(results):
    """Extract hand landmarks as a fixed-size feature vector"""
    all_points = []
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                all_points.extend([lm.x, lm.y, lm.z])
    
    features = np.array(all_points)
    
    # Pad or truncate to fixed size (126)
    fixed_size = 126
    if len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    elif len(features) > fixed_size:
        features = features[:fixed_size]
    
    return features

def smooth_predictions(prediction_history, current_pred, threshold=10):
    """Smooth predictions using majority voting with balanced threshold"""
    prediction_history.append(current_pred)
    if len(prediction_history) < threshold:
        return current_pred
    
    from collections import Counter
    counts = Counter(list(prediction_history))
    most_common = counts.most_common(1)[0]
    
    # Require 60% majority for balanced accuracy and responsiveness
    if most_common[1] >= len(prediction_history) * 0.60:
        return most_common[0]
    
    return current_pred

# ---------------------------------------------------------
# VIDEO FEED GENERATOR
# ---------------------------------------------------------
def generate_frames():
    """Generate video frames with sign language detection"""
    global detected_text, word_builder, last_alphabet, last_time, current_mode
    global alphabet_history, sentence_history, stable_alphabet, stable_count
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        alphabet_text = ""
        sentence_text = ""
        confidence = 0.0
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
            
            # Extract features and predict
            if alphabet_model and sentence_model:
                features = extract_landmarks(results)
                
                if current_mode == 'alphabet':
                    pred_idx = alphabet_model.predict([features])[0]
                    raw_alphabet = alphabet_labels.inverse_transform([pred_idx])[0]
                    
                    # Get confidence
                    try:
                        proba = alphabet_model.predict_proba([features])[0]
                        confidence = np.max(proba) * 100
                    except:
                        confidence = 0.0
                    
                    # Only accept predictions with sufficient confidence
                    if confidence >= min_confidence * 100:
                        alphabet_text = smooth_predictions(alphabet_history, raw_alphabet)
                    else:
                        alphabet_text = ""
                    
                    detected_text = alphabet_text if alphabet_text else "-"
                    
                    # ---------------------------
                    # LETTER STABILITY CHECK
                    # ---------------------------
                    if alphabet_text:  # only if we have a valid prediction
                        
                        if alphabet_text == stable_alphabet:
                            stable_count += 1
                        else:
                            stable_alphabet = alphabet_text
                            stable_count = 1

                        # When letter is stable enough → add to word_builder
                        if stable_count >= required_stability:
                            if len(word_builder) == 0 or word_builder[-1] != alphabet_text:
                                word_builder.append(alphabet_text)
                                print(f"✅ Auto-added letter: {alphabet_text}")

                            stable_count = 0  # reset for next letter
                    else:
                        # Reset stability if no valid detection
                        stable_count = 0
                        
                else:  # sentence mode
                    pred_idx = sentence_model.predict([features])[0]
                    sentence_text = sentence_labels.inverse_transform([pred_idx])[0]
                    sentence_text = smooth_predictions(sentence_history, sentence_text)
                    detected_text = sentence_text.replace('_', ' ')
                    
                    try:
                        proba = sentence_model.predict_proba([features])[0]
                        confidence = np.max(proba) * 100
                    except:
                        confidence = 0.0
        
        # Add detection info overlay
        cv2.rectangle(frame, (0, 0), (w, 80), (50, 50, 50), -1)
        
        mode_color = (0, 255, 255) if current_mode == 'alphabet' else (255, 100, 255)
        mode_text = f"Mode: {current_mode.upper()}"
        cv2.putText(frame, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        if detected_text:
            # Show detected letter
            cv2.putText(frame, f"Detected: {detected_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Show confidence score
            if confidence > 0:
                conf_color = (0, 255, 0) if confidence >= 60 else (0, 255, 255) if confidence >= 50 else (0, 165, 255)
                cv2.putText(frame, f"Conf: {confidence:.1f}%", (w-150, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Hand detection indicator
        if results.multi_hand_landmarks:
            cv2.circle(frame, (w-30, 30), 12, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (w-30, 30), 12, (0, 0, 255), -1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', models_loaded=(alphabet_model is not None))

@app.route('/tutorials')
def tutorials():
    """Tutorial videos page"""
    return render_template('tutorials.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection')
def get_detection():
    """Get current detection results"""
    global detected_text, word_builder, current_mode
    
    built_word = ''.join(word_builder) if current_mode == 'alphabet' else ""
    
    return jsonify({
        'mode': current_mode,
        'detected': detected_text,
        'word': built_word,
        'success': True
    })

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Change detection mode"""
    global current_mode, alphabet_history, sentence_history
    
    data = request.json
    new_mode = data.get('mode', 'alphabet')
    
    if new_mode in ['alphabet', 'sentence']:
        current_mode = new_mode
        alphabet_history.clear()
        sentence_history.clear()
        return jsonify({'success': True, 'mode': current_mode})
    
    return jsonify({'success': False, 'error': 'Invalid mode'})

@app.route('/add_letter', methods=['POST'])
def add_letter():
    """Manually add the currently detected letter to word"""
    global word_builder, detected_text
    
    if detected_text and detected_text != "-" and detected_text.strip():
        # Don't add duplicate letters consecutively
        if len(word_builder) == 0 or word_builder[-1] != detected_text:
            word_builder.append(detected_text)
            return jsonify({
                'success': True, 
                'word': ''.join(word_builder),
                'letter': detected_text
            })
    
    return jsonify({
        'success': False, 
        'error': 'No letter detected',
        'word': ''.join(word_builder)
    })

@app.route('/add_space', methods=['POST'])
def add_space():
    """Add space to word"""
    global word_builder
    word_builder.append(' ')
    return jsonify({'success': True, 'word': ''.join(word_builder)})

@app.route('/backspace', methods=['POST'])
def backspace():
    """Remove last character"""
    global word_builder
    if word_builder:
        word_builder.pop()
    return jsonify({'success': True, 'word': ''.join(word_builder)})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    """Clear all text"""
    global word_builder, detected_text
    word_builder.clear()
    detected_text = ""
    return jsonify({'success': True})

# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == '__main__':
    if error:
        print(f"⚠️ Warning: Could not load models: {error}")
        print("The app will run but detection won't work until models are trained.")
    else:
        print("✅ Models loaded successfully!")
    
    print("\n" + "="*60)
    print("🇮🇳 ISL (Indian Sign Language) Recognition App")
    print("="*60)
    print("📱 Open browser: http://localhost:5000")
    print("📹 ISL Reference: https://www.youtube.com/watch?v=qcdivQfA41Y")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
