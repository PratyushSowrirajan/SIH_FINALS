import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

class ExpressionMLTrainer:
    """
    ML-based expression trainer that collects samples and trains a classifier.
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Expression labels
        self.expressions = ['happy', 'sad', 'angry', 'surprised', 'sleepy']
        self.all_samples = []
        self.all_labels = []
        
    def extract_features(self, landmarks):
        """Extract comprehensive features from face landmarks."""
        features = []
        
        # Key landmark indices
        left_eye = [33, 160, 158, 133, 153, 144, 145, 159]
        right_eye = [362, 385, 387, 263, 373, 380, 374, 386]
        mouth = [61, 291, 0, 17, 13, 14, 78, 308, 95, 88]
        left_eyebrow = [70, 63, 105, 66, 107]
        right_eyebrow = [336, 296, 334, 293, 300]
        nose = [1, 2, 98, 327, 6]
        
        # Distance-based features
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        # Eye features
        features.append(dist(159, 145))  # Left eye height
        features.append(dist(386, 374))  # Right eye height
        features.append(dist(33, 133))   # Left eye width
        features.append(dist(362, 263))  # Right eye width
        features.append(dist(133, 263))  # Inter-eye distance
        
        # Mouth features
        features.append(dist(13, 14))    # Mouth height
        features.append(dist(61, 291))   # Mouth width
        features.append(dist(0, 17))     # Mouth opening
        features.append(dist(78, 308))   # Upper lip width
        features.append(dist(95, 88))    # Lower lip width
        
        # Eyebrow features
        features.append(dist(70, 10))    # Left eyebrow to nose
        features.append(dist(336, 10))   # Right eyebrow to nose
        features.append(dist(105, 159))  # Left eyebrow to eye
        features.append(dist(334, 386))  # Right eyebrow to eye
        
        # Nose features
        features.append(dist(1, 2))      # Nose bridge
        features.append(dist(98, 327))   # Nose width
        
        # Face proportions
        face_height = dist(10, 152)
        face_width = dist(234, 454)
        features.append(face_height)
        features.append(face_width)
        
        # Ratios
        if face_width > 0:
            features.append(face_height / face_width)
        else:
            features.append(0)
        
        mouth_width = dist(61, 291)
        if mouth_width > 0:
            features.append(dist(13, 14) / mouth_width)  # Mouth aspect ratio
        else:
            features.append(0)
        
        # Relative positions (normalized by face size)
        for idx in left_eye + right_eye + [61, 291, 13, 14]:
            features.append(landmarks[idx][0] / (face_width + 1e-6))
            features.append(landmarks[idx][1] / (face_height + 1e-6))
        
        return np.array(features)
    
    def collect_expression_samples(self, expression_name, num_samples=200):
        """Collect samples for one expression."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Create fullscreen window
        cv2.namedWindow('Expression Training', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Expression Training', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print(f"\n{'='*60}")
        print(f"Collecting samples for: {expression_name.upper()}")
        print(f"Target: {num_samples} frames")
        print("Press SPACE to start collecting")
        print("Make the expression and hold it!")
        print(f"{'='*60}\n")
        
        samples = []
        collecting = False
        
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Draw thin white mesh with multiple connection sets for more nodes
                thin_mesh = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(255, 255, 255))
                
                # Draw tesselation (main mesh)
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    None, thin_mesh
                )
                
                # Draw contours for more detail and nodes
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, thin_mesh
                )
                
                # Draw irises for eye detail
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    self.mp_face_mesh.FACEMESH_IRISES,
                    None, thin_mesh
                )
                
                if collecting:
                    landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark])
                    features = self.extract_features(landmarks)
                    samples.append(features)
            
            # Draw clean professional UI
            progress = len(samples) / num_samples
            
            # Clean overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Frame border
            cv2.rectangle(frame, (0, 0), (w, 120), (255, 255, 255), 2)
            
            # Expression title - clean white
            cv2.putText(frame, f"TRAINING: {expression_name.upper()}", 
                       (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            if not collecting:
                # Simple instruction
                cv2.putText(frame, "PRESS SPACE TO START COLLECTING", 
                           (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                # Progress text - simple and clean
                cv2.putText(frame, f"PROGRESS: {len(samples)}/{num_samples} ({progress*100:.0f}%)", 
                           (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Progress bar
                bar_width = w - 100
                bar_x, bar_y = 50, h - 80
                bar_height = 40
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                filled = int(bar_width * progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 255, 0), -1)
            
            cv2.imshow('Expression Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and not collecting:  # SPACE
                collecting = True
                print(f"Collecting {expression_name}...")
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(samples) >= num_samples * 0.8:
            print(f"✓ Collected {len(samples)} samples for {expression_name}")
            return samples
        else:
            print(f"✗ Insufficient samples: {len(samples)}/{num_samples}")
            return None
    
    def train_all_expressions(self):
        """Collect samples for all expressions and train the model."""
        print("\n" + "="*60)
        print("ML EXPRESSION TRAINING")
        print("="*60)
        print("\nWe will collect 200 samples for each expression:")
        for expr in self.expressions:
            print(f"  - {expr.upper()}")
        print("\nTotal: 1000 samples")
        print("="*60)
        
        input("\nPress Enter to start...")
        
        # Collect samples for each expression
        for expr in self.expressions:
            samples = self.collect_expression_samples(expr, 200)
            if samples is not None:
                self.all_samples.extend(samples)
                self.all_labels.extend([expr] * len(samples))
            else:
                print(f"Failed to collect samples for {expr}")
                return False
        
        # Train the model
        print("\n" + "="*60)
        print("Training ML Model...")
        print("="*60)
        
        X = np.array(self.all_samples)
        y = np.array(self.all_labels)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # Calculate accuracy
        accuracy = self.model.score(X_scaled, y)
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Training accuracy: {accuracy*100:.2f}%")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        
        # Save model
        os.makedirs('ml_models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'expressions': self.expressions,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'num_samples': len(X),
            'num_features': X.shape[1]
        }
        
        model_path = 'ml_models/expression_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved: {model_path}")
        print("="*60 + "\n")
        
        return True


def main():
    trainer = ExpressionMLTrainer()
    
    if trainer.train_all_expressions():
        print("\n✓ Training complete! Run face_ml_detector.py to test.")
    else:
        print("\n✗ Training failed.")


if __name__ == '__main__':
    main()
