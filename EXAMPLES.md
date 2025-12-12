# Usage Examples

This document shows different ways to use the Facial Expression Recognition System.

## Basic Usage

### Run the Main Application
```bash
python main.py
```

The app will:
1. Open your webcam
2. Display detected expressions in real-time
3. Show confidence scores
4. Update 25-30 times per second

**Controls:**
- **Q** - Quit
- **C** - Configure new expression
- **S** - Save screenshot

---

## Calibrating Custom Expressions

### Interactive Calibrator

Run the calibration tool:
```bash
python configure_expressions.py
```

**Menu Options:**

#### 1. Calibrate New Expression
- Collects 15 samples of an expression
- Automatically generates optimal thresholds
- Saves to `expressions.json`

Steps:
1. Choose option 1
2. Enter expression name (e.g., "winking")
3. Make the expression
4. Press SPACE to capture (15 times)
5. System calculates thresholds

#### 2. View Configured Expressions
Shows all expressions in `expressions.json`.

#### 3. Delete Expression
Remove a custom expression.

#### 4. Test Detection
Live test of expression detection.

---

## Advanced Configuration

### Manually Edit expressions.json

Edit the configuration file directly:

```json
{
  "confused": {
    "mouth_openness": [0.05, 0.20],
    "mouth_aspect_ratio": [0.02, 0.10],
    "avg_eyebrow_raise": [0.10, 0.25],
    "left_eyebrow_raise": [0.08, 0.30],
    "right_eyebrow_raise": [0.08, 0.30]
  },
  
  "shouting": {
    "mouth_openness": [0.30, 0.70],
    "mouth_width": [0.4, 1.0],
    "avg_eye_openness": [0.15, 0.35],
    "avg_eyebrow_raise": [0.15, 0.35],
    "nostril_flare": [0.10, 0.18]
  },
  
  "skeptical": {
    "mouth_openness": [0.05, 0.15],
    "mouth_aspect_ratio": [0.01, 0.08],
    "left_eyebrow_raise": [0.15, 0.35],
    "right_eyebrow_raise": [-0.10, 0.05]
  }
}
```

---

## Programmatic Usage

### Use in Your Own Python Code

#### Example 1: Basic Expression Detection

```python
from expression_detector import ExpressionDetector
from main import FacialExpressionRecognizer
import cv2

# Initialize recognizer
recognizer = FacialExpressionRecognizer('expressions.json')

# Get webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    frame, expression, confidence = recognizer.process_frame(frame)
    
    # Your custom logic here
    if expression == 'happy' and confidence > 0.7:
        print("User is happy!")
    elif expression == 'sad' and confidence > 0.7:
        print("User is sad!")
    
    cv2.imshow('Expression', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Example 2: Detect and Log Expressions

```python
import json
import csv
from datetime import datetime
from main import FacialExpressionRecognizer

recognizer = FacialExpressionRecognizer('expressions.json')
expression_log = []

cap = cv2.VideoCapture(0)
frame_count = 0

while frame_count < 300:  # Record for 10 seconds at 30 FPS
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, expression, confidence = recognizer.process_frame(frame)
    
    # Log expression
    expression_log.append({
        'timestamp': datetime.now().isoformat(),
        'expression': expression,
        'confidence': confidence,
        'frame': frame_count
    })
    
    frame_count += 1

# Save to CSV
with open('expression_log.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'expression', 'confidence', 'frame'])
    writer.writeheader()
    writer.writerows(expression_log)

cap.release()
print(f"Logged {len(expression_log)} expressions")
```

#### Example 3: Real-time Expression Statistics

```python
from collections import Counter
from main import FacialExpressionRecognizer

recognizer = FacialExpressionRecognizer('expressions.json')
expression_counter = Counter()

cap = cv2.VideoCapture(0)
frame_count = 0

while frame_count < 600:  # 20 seconds at 30 FPS
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, expression, confidence = recognizer.process_frame(frame)
    
    # Only count high-confidence detections
    if confidence > 0.6:
        expression_counter[expression] += 1
    
    # Print stats every 10 seconds
    if frame_count % 300 == 0 and frame_count > 0:
        print(f"\n=== Statistics at {frame_count/30:.0f}s ===")
        for expr, count in expression_counter.most_common(5):
            percentage = (count / (frame_count/30)) * 100
            print(f"{expr}: {count} times ({percentage:.1f}%)")
    
    frame_count += 1

cap.release()

# Final stats
print("\n=== Final Statistics ===")
for expr, count in expression_counter.most_common(10):
    percentage = (count / (frame_count/30)) * 100
    print(f"{expr}: {count} ({percentage:.1f}%)")
```

#### Example 4: Conditional Actions Based on Expression

```python
from main import FacialExpressionRecognizer
import winsound  # Windows only

recognizer = FacialExpressionRecognizer('expressions.json')

def on_expression_detected(expression, confidence):
    """Execute action when expression detected."""
    
    if expression == 'happy' and confidence > 0.75:
        print("😊 Smile detected!")
        # Play sound, send notification, etc.
        winsound.Beep(1000, 200)
    
    elif expression == 'surprised' and confidence > 0.75:
        print("😲 Surprise detected!")
        winsound.Beep(2000, 200)
    
    elif expression == 'angry' and confidence > 0.75:
        print("😠 Anger detected!")
        winsound.Beep(500, 300)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, expression, confidence = recognizer.process_frame(frame)
    on_expression_detected(expression, confidence)
    
    frame = recognizer.draw_ui(frame, expression, confidence)
    cv2.imshow('Expression Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Integration with Other Systems

### Example 1: Flask Web Server

```python
from flask import Flask, render_template, Response
from main import FacialExpressionRecognizer
import cv2

app = Flask(__name__)
recognizer = FacialExpressionRecognizer('expressions.json')
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame, expression, confidence = recognizer.process_frame(frame)
        frame = recognizer.draw_ui(frame, expression, confidence)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
```

### Example 2: OpenCV Window with Custom Processing

```python
from main import FacialExpressionRecognizer
import cv2

recognizer = FacialExpressionRecognizer('expressions.json')
cap = cv2.VideoCapture(0)

expression_timeline = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process
    frame, expression, confidence = recognizer.process_frame(frame)
    
    # Draw custom UI
    h, w, _ = frame.shape
    
    # Expression timeline at bottom
    timeline_y = h - 50
    for i, (prev_expr, prev_conf) in enumerate(expression_timeline[-10:]):
        color = recognizer.colors.get(prev_expr, (200, 200, 200))
        x = 50 + i * 40
        cv2.rectangle(frame, (x, timeline_y), (x+35, timeline_y+30), color, -1)
    
    # Main UI
    frame = recognizer.draw_ui(frame, expression, confidence)
    
    # Store in timeline
    expression_timeline.append((expression, confidence))
    if len(expression_timeline) > 20:
        expression_timeline.pop(0)
    
    cv2.imshow('Expression Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Troubleshooting Examples

### Problem: Expression Not Being Detected

```python
from expression_detector import ExpressionDetector
import numpy as np

detector = ExpressionDetector()

# Check extracted features
features = detector.extract_features(landmarks)
print("Extracted Features:")
for feature, value in features.items():
    print(f"  {feature}: {value:.4f}")

# Check expression definitions
default_exprs = detector._get_default_expressions()
happy_thresholds = default_exprs['happy']
print("\nHappy thresholds:")
for feature, (min_v, max_v) in happy_thresholds.items():
    print(f"  {feature}: {min_v:.4f} - {max_v:.4f}")

# Calculate match score
score = detector._calculate_match_score(features, happy_thresholds)
print(f"\nMatch score: {score:.2f}")
```

### Problem: Slow Performance

```python
import time
from main import FacialExpressionRecognizer

recognizer = FacialExpressionRecognizer('expressions.json')
cap = cv2.VideoCapture(0)

# Set for optimal speed
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = []
start = time.time()

for i in range(30):  # Measure 30 frames
    frame_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, expr, conf = recognizer.process_frame(frame)
    
    frame_time = time.time() - frame_start
    frame_times.append(frame_time * 1000)  # Convert to ms

cap.release()

print(f"Average frame time: {np.mean(frame_times):.2f}ms")
print(f"Min: {np.min(frame_times):.2f}ms, Max: {np.max(frame_times):.2f}ms")
print(f"FPS: {1000/np.mean(frame_times):.1f}")
```

---

## Testing and Validation

### Test All Expressions

```bash
python configure_expressions.py
# Select option 4: Test expression detection
```

### Batch Test with Multiple Faces

1. Record video of different people
2. Extract frames every 100ms
3. Run expression detection
4. Compare results

---

## Performance Tips

1. **Lower Resolution** - Set 640x480 instead of 1920x1080
2. **Reduce Landmarks** - Draw every 3rd instead of all
3. **Skip Frames** - Process every 2nd frame if latency is critical
4. **Close Apps** - Reduces CPU contention
5. **GPU** - MediaPipe uses GPU automatically if available

---

## Next Steps

- Start with `python main.py`
- Calibrate custom expressions with `python configure_expressions.py`
- Integrate into your application using the examples above
- Adjust thresholds in `expressions.json` for better accuracy

See `README.md` and `IMPLEMENTATION.md` for more details.
