import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import time

class ExpressionSystem:
    """
    Complete Expression Recognition System with collect, train, and predict modes.
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 5 expressions mapped to number keys
        self.expressions = {
            1: 'happy',
            2: 'sad',
            3: 'angry',
            4: 'surprised',
            5: 'sleepy'
        }
        
        self.samples_dir = 'expression_samples'
        self.model_path = 'ml_models/expression_model.pkl'
        
        # Create directories
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs('ml_models', exist_ok=True)
    
    def extract_features(self, landmarks):
        """Extract comprehensive facial features."""
        features = []
        
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        def angle(p1, p2, p3):
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.arccos(np.clip(cos_angle, -1, 1))
        
        # Eye features (8 measurements)
        features.append(dist(159, 145))  # Left eye height
        features.append(dist(386, 374))  # Right eye height
        features.append(dist(33, 133))   # Left eye width
        features.append(dist(362, 263))  # Right eye width
        features.append(dist(133, 263))  # Eye distance
        features.append(dist(145, 374))  # Eye-to-eye vertical
        features.append(dist(159, 386))  # Outer eye corners
        features.append(dist(145, 159))  # Left eye aspect
        
        # Mouth features (12 measurements)
        features.append(dist(13, 14))    # Mouth height
        features.append(dist(61, 291))   # Mouth width
        features.append(dist(0, 17))     # Upper-lower lip
        features.append(dist(78, 308))   # Mouth corners width
        features.append(dist(95, 88))    # Inner mouth height
        features.append(dist(13, 312))   # Upper lip to top
        features.append(dist(14, 178))   # Lower lip to bottom
        features.append(dist(61, 62))    # Left mouth corner
        features.append(dist(291, 292))  # Right mouth corner
        features.append(dist(37, 267))   # Mouth width outer
        features.append(dist(82, 13))    # Left upper lip
        features.append(dist(312, 13))   # Right upper lip
        
        # Eyebrow features (8 measurements)
        features.append(dist(70, 63))    # Left eyebrow length
        features.append(dist(336, 296))  # Right eyebrow length
        features.append(dist(105, 159))  # Left eyebrow to eye
        features.append(dist(334, 386))  # Right eyebrow to eye
        features.append(dist(107, 66))   # Left eyebrow inner
        features.append(dist(336, 296))  # Right eyebrow inner
        features.append(dist(70, 107))   # Left eyebrow span
        features.append(dist(336, 300))  # Right eyebrow span
        
        # Nose features (6 measurements)
        features.append(dist(1, 2))      # Nose bridge
        features.append(dist(98, 327))   # Nose width
        features.append(dist(4, 5))      # Nose tip
        features.append(dist(195, 197))  # Nose bottom
        features.append(dist(168, 6))    # Nose height
        features.append(dist(168, 197))  # Nose vertical
        
        # Face geometry (6 measurements)
        face_height = dist(10, 152)
        face_width = dist(234, 454)
        features.append(face_height)
        features.append(face_width)
        features.append(dist(234, 10))   # Left face diagonal
        features.append(dist(454, 10))   # Right face diagonal
        features.append(dist(234, 152))  # Left jaw diagonal
        features.append(dist(454, 152))  # Right jaw diagonal
        
        # Ratios (10 measurements)
        features.append(face_height / (face_width + 1e-6))
        features.append(dist(13, 14) / (dist(61, 291) + 1e-6))  # Mouth aspect
        features.append(dist(159, 145) / (dist(33, 133) + 1e-6))  # Left eye aspect
        features.append(dist(386, 374) / (dist(362, 263) + 1e-6))  # Right eye aspect
        features.append(dist(105, 159) / (face_height + 1e-6))  # Eyebrow height ratio
        features.append(dist(13, 14) / (face_height + 1e-6))  # Mouth height ratio
        features.append(dist(61, 291) / (face_width + 1e-6))  # Mouth width ratio
        features.append(dist(98, 327) / (face_width + 1e-6))  # Nose width ratio
        features.append(dist(133, 263) / (face_width + 1e-6))  # Eye distance ratio
        features.append((dist(159, 145) + dist(386, 374)) / (dist(13, 14) + 1e-6))  # Eye-mouth ratio
        
        # Angles (6 measurements)
        features.append(angle(70, 63, 105))   # Left eyebrow angle
        features.append(angle(336, 296, 334))  # Right eyebrow angle
        features.append(angle(61, 13, 291))    # Mouth smile angle
        features.append(angle(33, 159, 133))   # Left eye angle
        features.append(angle(362, 386, 263))  # Right eye angle
        features.append(angle(1, 4, 152))      # Nose angle
        
        # Normalized positions (12 measurements for key landmarks)
        key_points = [10, 152, 234, 454, 159, 386, 13, 14, 61, 291, 70, 336]
        for idx in key_points:
            features.append(landmarks[idx][1] / (face_height + 1e-6))
        
        return np.array(features)
    
    def draw_mesh(self, frame, face_landmarks):
        """Draw clean white mesh."""
        thin_mesh = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(255, 255, 255))
        
        self.mp_drawing.draw_landmarks(
            frame, face_landmarks,
            self.mp_face_mesh.FACEMESH_TESSELATION,
            None, thin_mesh
        )
        
        self.mp_drawing.draw_landmarks(
            frame, face_landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            None, thin_mesh
        )
        
        self.mp_drawing.draw_landmarks(
            frame, face_landmarks,
            self.mp_face_mesh.FACEMESH_IRISES,
            None, thin_mesh
        )
    
    def collect_samples(self):
        """Collect 200 frame samples for each expression using number keys."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cv2.namedWindow('Collect Expression Samples', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Collect Expression Samples', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\n" + "="*70)
        print("DATA COLLECTION MODE")
        print("="*70)
        print("Press number keys to collect samples:")
        print("  1 = HAPPY")
        print("  2 = SAD")
        print("  3 = ANGRY")
        print("  4 = SURPRISED")
        print("  5 = DISGUSTED")
        print("\nEach press collects 200 frames (takes ~7 seconds)")
        print("Press ESC to exit")
        print("="*70 + "\n")
        
        collecting = False
        current_expression = None
        samples = []
        target_samples = 200
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                self.draw_mesh(frame, face_landmarks)
                
                if collecting:
                    landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark])
                    features = self.extract_features(landmarks)
                    samples.append(features)
                    
                    if len(samples) >= target_samples:
                        # Save samples
                        filename = f"{self.samples_dir}/{self.expressions[current_expression]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
                        np.save(filename, np.array(samples))
                        print(f"✓ Saved {len(samples)} samples for {self.expressions[current_expression].upper()}")
                        
                        collecting = False
                        samples = []
            
            # UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.rectangle(frame, (0, 0), (w, 150), (255, 255, 255), 2)
            
            if collecting:
                progress = len(samples) / target_samples
                cv2.putText(frame, f"COLLECTING: {self.expressions[current_expression].upper()}", 
                           (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, f"PROGRESS: {len(samples)}/{target_samples} ({progress*100:.0f}%)", 
                           (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "PRESS 1-5 TO COLLECT EXPRESSION", 
                           (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, "1=HAPPY  2=SAD  3=ANGRY  4=SURPRISED  5=SLEEPY", 
                           (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Collect Expression Samples', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')] and not collecting:
                current_expression = int(chr(key))
                collecting = True
                samples = []
                print(f"Collecting {self.expressions[current_expression].upper()}...")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def train_model(self):
        """Train model from collected samples."""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        # Load all samples
        all_samples = []
        all_labels = []
        
        for expr_num, expr_name in self.expressions.items():
            files = [f for f in os.listdir(self.samples_dir) if f.startswith(expr_name) and f.endswith('.npy')]
            
            if not files:
                print(f"✗ No samples found for {expr_name.upper()}")
                continue
            
            for file in files:
                samples = np.load(os.path.join(self.samples_dir, file))
                all_samples.extend(samples)
                all_labels.extend([expr_name] * len(samples))
                print(f"  Loaded {len(samples)} samples from {file}")
        
        if len(all_samples) == 0:
            print("\n✗ No training data found! Collect samples first.")
            return False
        
        print(f"\nTotal samples: {len(all_samples)}")
        
        # Train
        X = np.array(all_samples)
        y = np.array(all_labels)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        accuracy = model.score(X_scaled, y)
        print(f"\nTraining Accuracy: {accuracy*100:.2f}%")
        
        # Save
        model_data = {
            'model': model,
            'scaler': scaler,
            'expressions': list(self.expressions.values()),
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(all_samples)
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {self.model_path}")
        print("="*70 + "\n")
        return True
    
    def predict(self):
        """Run real-time prediction."""
        if not os.path.exists(self.model_path):
            print("\n✗ Model not found! Train model first.")
            return
        
        # Load model
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        accuracy = model_data['accuracy']
        
        print("\n" + "="*70)
        print("PREDICTION MODE")
        print("="*70)
        print(f"Model Accuracy: {accuracy*100:.2f}%")
        print(f"Expressions: {', '.join([e.upper() for e in self.expressions.values()])}")
        print("Press 'Q' to exit")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cv2.namedWindow('Expression Prediction', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Expression Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        prediction_history = deque(maxlen=5)
        fps_history = deque(maxlen=30)
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            expression = 'neutral'
            confidence = 0.0
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                self.draw_mesh(frame, face_landmarks)
                
                landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark])
                features = self.extract_features(landmarks)
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
                
                if confidence >= 0.5:
                    expression = prediction
                    prediction_history.append(prediction)
                    
                    # Use mode for stability
                    if len(prediction_history) >= 3:
                        from collections import Counter
                        expression = Counter(prediction_history).most_common(1)[0][0]
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time + 1e-6)
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.rectangle(frame, (0, 0), (w, 100), (255, 255, 255), 2)
            
            cv2.putText(frame, f"EXPRESSION: {expression.upper()}", 
                       (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"CONFIDENCE: {confidence*100:.0f}%", 
                       (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.rectangle(frame, (0, h-60), (w, h), (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {avg_fps:.0f}", 
                       (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MODEL ACCURACY: {accuracy*100:.0f}%", 
                       (w//2 - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "PRESS 'Q' TO EXIT", 
                       (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Expression Prediction', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    system = ExpressionSystem()
    
    while True:
        print("\n" + "="*70)
        print("FACIAL EXPRESSION RECOGNITION SYSTEM")
        print("="*70)
        print("1. COLLECT DATA (Press 1-5 for expressions, 200 frames each)")
        print("2. TRAIN MODEL (Train from collected samples)")
        print("3. PREDICT (Run real-time detection)")
        print("4. EXIT")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            system.collect_samples()
        elif choice == '2':
            system.train_model()
        elif choice == '3':
            system.predict()
        elif choice == '4':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice!")


if __name__ == '__main__':
    main()
