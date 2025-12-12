import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

class AccurateExpressionTrainer:
    """
    High-accuracy expression trainer with number key controls.
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Map keys to expressions
        self.key_to_expression = {
            ord('1'): 'happy',
            ord('2'): 'sad', 
            ord('3'): 'angry',
            ord('4'): 'surprised',
            ord('5'): 'disgusted'
        }
        
        self.expressions = list(self.key_to_expression.values())
        self.all_samples = []
        self.all_labels = []
        
    def extract_features(self, landmarks):
        """Extract 60+ highly accurate features."""
        features = []
        
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        def angle(p1, p2, p3):
            """Calculate angle between three points."""
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.arccos(np.clip(cosine, -1.0, 1.0))
        
        # Face dimensions
        face_height = dist(10, 152)
        face_width = dist(234, 454)
        
        # EYE FEATURES (Critical for expressions)
        # Left eye
        features.append(dist(159, 145))  # height
        features.append(dist(33, 133))   # width
        features.append(dist(159, 145) / (dist(33, 133) + 1e-6))  # aspect ratio
        # Right eye
        features.append(dist(386, 374))  # height
        features.append(dist(362, 263))  # width
        features.append(dist(386, 374) / (dist(362, 263) + 1e-6))  # aspect ratio
        # Eye positions
        features.append(dist(133, 263))  # inter-eye distance
        features.append((landmarks[159][1] + landmarks[386][1]) / 2 / face_height)  # avg eye height
        
        # MOUTH FEATURES (Critical for expressions)
        features.append(dist(13, 14))    # mouth height (vertical opening)
        features.append(dist(61, 291))   # mouth width
        features.append(dist(13, 14) / (dist(61, 291) + 1e-6))  # mouth aspect ratio
        features.append(dist(0, 17))     # inner mouth opening
        features.append(dist(78, 308))   # upper lip span
        features.append(dist(95, 88))    # lower lip span
        # Mouth corners
        features.append(landmarks[61][1] / face_height)   # left corner vertical position
        features.append(landmarks[291][1] / face_height)  # right corner vertical position
        # Lip curvature
        features.append(angle(61, 13, 291))  # upper lip angle
        features.append(angle(61, 14, 291))  # lower lip angle
        
        # EYEBROW FEATURES (Critical for expressions)
        # Left eyebrow
        features.append(dist(70, 159))   # eyebrow to eye distance
        features.append(dist(63, 159))
        features.append(dist(105, 159))
        # Right eyebrow
        features.append(dist(336, 386))  # eyebrow to eye distance
        features.append(dist(296, 386))
        features.append(dist(334, 386))
        # Eyebrow positions
        features.append(landmarks[70][1] / face_height)   # left eyebrow height
        features.append(landmarks[336][1] / face_height)  # right eyebrow height
        # Eyebrow angles
        features.append(angle(70, 63, 105))   # left eyebrow angle
        features.append(angle(336, 296, 334)) # right eyebrow angle
        
        # NOSE FEATURES
        features.append(dist(1, 2))      # nose bridge length
        features.append(dist(98, 327))   # nose width
        features.append(dist(4, 5))      # nose tip width
        
        # CHEEK FEATURES
        features.append(dist(50, 280))   # cheek span
        features.append(dist(205, 425))  # mid-face width
        
        # JAW/CHIN FEATURES
        features.append(dist(10, 152))   # jaw to chin
        features.append(dist(172, 397))  # jaw width
        
        # COMPREHENSIVE RATIOS
        features.append(face_height / (face_width + 1e-6))  # face aspect ratio
        features.append(dist(13, 14) / (face_height + 1e-6))  # mouth height ratio
        features.append(dist(61, 291) / (face_width + 1e-6))  # mouth width ratio
        features.append(dist(159, 145) / (face_height + 1e-6))  # left eye height ratio
        features.append(dist(386, 374) / (face_height + 1e-6))  # right eye height ratio
        
        # DETAILED LANDMARK POSITIONS (normalized)
        key_points = [
            # Eye corners
            33, 133, 362, 263,
            # Mouth corners and center
            61, 291, 0, 17, 13, 14,
            # Eyebrow points
            70, 63, 105, 336, 296, 334,
            # Nose
            1, 2, 4, 5,
            # Face outline
            10, 152, 234, 454
        ]
        
        for idx in key_points:
            features.append(landmarks[idx][0] / (face_width + 1e-6))   # x normalized
            features.append(landmarks[idx][1] / (face_height + 1e-6))  # y normalized
        
        return np.array(features)
    
    def train_by_keys(self):
        """Train by pressing number keys 1-5."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cv2.namedWindow('Expression Training', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Expression Training', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Storage for each expression
        expression_samples = {expr: [] for expr in self.expressions}
        current_expression = None
        target_samples = 200
        
        print("\n" + "="*80)
        print("ACCURATE EXPRESSION TRAINING")
        print("="*80)
        print("\nKEY MAPPING:")
        print("  [1] = HAPPY       (Big smile, raised cheeks)")
        print("  [2] = SAD         (Frown, droopy eyes)")
        print("  [3] = ANGRY       (Furrowed brows, tight lips)")
        print("  [4] = SURPRISED   (Wide eyes, open mouth)")
        print("  [5] = DISGUSTED   (Wrinkled nose, raised upper lip)")
        print("\nINSTRUCTIONS:")
        print("  - Press number key 1-5 to capture that expression")
        print("  - Make EXAGGERATED expressions for better accuracy")
        print("  - Hold expression steady while capturing")
        print("  - Collect 200 samples per expression")
        print("  - Press 'Q' to finish and train model")
        print("="*80 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            landmarks = None
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Draw thin white mesh
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
                
                landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark])
            
            # Draw UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.rectangle(frame, (0, 0), (w, 200), (255, 255, 255), 2)
            
            # Title
            cv2.putText(frame, "PRESS NUMBER KEY (1-5) TO CAPTURE EXPRESSION", 
                       (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Expression status
            y_offset = 80
            for i, expr in enumerate(self.expressions, 1):
                count = len(expression_samples[expr])
                status = f"[{i}] {expr.upper()}: {count}/{target_samples}"
                color = (255, 255, 255) if count >= target_samples else (180, 180, 180)
                cv2.putText(frame, status, 
                           (30 + (i-1)*320, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Instructions
            cv2.putText(frame, "PRESS 'Q' TO FINISH AND TRAIN MODEL", 
                       (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            cv2.imshow('Expression Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture on number keys
            if key in self.key_to_expression and landmarks is not None:
                expr = self.key_to_expression[key]
                if len(expression_samples[expr]) < target_samples:
                    features = self.extract_features(landmarks)
                    expression_samples[expr].append(features)
                    print(f"✓ Captured {expr}: {len(expression_samples[expr])}/{target_samples}")
            
            # Finish
            elif key == ord('q'):
                # Check if enough samples
                total = sum(len(samples) for samples in expression_samples.values())
                if total >= 500:  # At least 100 per expression on average
                    break
                else:
                    print(f"\n⚠ Need more samples! Total: {total}/1000 (minimum 500)")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Prepare training data
        for expr in self.expressions:
            for sample in expression_samples[expr]:
                self.all_samples.append(sample)
                self.all_labels.append(expr)
        
        if len(self.all_samples) < 100:
            print("\n✗ Insufficient samples collected!")
            return False
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL WITH {len(self.all_samples)} SAMPLES")
        print(f"{'='*80}")
        
        # Train model with optimized parameters
        X = np.array(self.all_samples)
        y = np.array(self.all_labels)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use more estimators for better accuracy
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining...")
        model.fit(X_scaled, y)
        
        # Cross-validation score
        print("\nEvaluating accuracy...")
        scores = cross_val_score(model, X_scaled, y, cv=5)
        accuracy = scores.mean()
        
        print(f"\n{'='*80}")
        print(f"✓ MODEL TRAINED SUCCESSFULLY!")
        print(f"  Cross-validation accuracy: {accuracy*100:.1f}%")
        print(f"  Training samples per expression:")
        for expr in self.expressions:
            count = len(expression_samples[expr])
            print(f"    {expr}: {count}")
        print(f"{'='*80}\n")
        
        # Save model
        os.makedirs('ml_models', exist_ok=True)
        model_path = 'ml_models/expression_model.pkl'
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'expressions': self.expressions,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'num_features': X.shape[1],
            'samples_per_expression': {expr: len(expression_samples[expr]) for expr in self.expressions}
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"\nNow run: python detect_expressions_ml.py")
        
        return True


def main():
    trainer = AccurateExpressionTrainer()
    trainer.train_by_keys()


if __name__ == '__main__':
    main()
