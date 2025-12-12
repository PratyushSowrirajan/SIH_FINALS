import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque
import time

class ExpressionMLDetector:
    """
    Real-time ML-based expression detection using trained model.
    """
    
    def __init__(self, model_path='ml_models/expression_model.pkl'):
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nPlease run train_expressions_ml.py first!")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.expressions = model_data['expressions']
        self.accuracy = model_data['accuracy']
        
        print(f"✓ Loaded model (accuracy: {self.accuracy*100:.2f}%)")
        print(f"  Expressions: {', '.join(self.expressions)}")
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Smoothing
        self.prediction_history = deque(maxlen=7)
        self.confidence_history = deque(maxlen=7)
        
        # Performance
        self.fps_history = deque(maxlen=30)
        
        # Colors
        self.colors = {
            'neutral': (200, 200, 200),
            'happy': (0, 255, 0),
            'sad': (255, 100, 100),
            'angry': (0, 0, 255),
            'surprised': (0, 165, 255),
            'disgusted': (128, 0, 128),
        }
    
    def extract_features(self, landmarks):
        """Extract same features as training."""
        features = []
        
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        # Eye features
        features.append(dist(159, 145))
        features.append(dist(386, 374))
        features.append(dist(33, 133))
        features.append(dist(362, 263))
        features.append(dist(133, 263))
        
        # Mouth features
        features.append(dist(13, 14))
        features.append(dist(61, 291))
        features.append(dist(0, 17))
        features.append(dist(78, 308))
        features.append(dist(95, 88))
        
        # Eyebrow features
        features.append(dist(70, 10))
        features.append(dist(336, 10))
        features.append(dist(105, 159))
        features.append(dist(334, 386))
        
        # Nose features
        features.append(dist(1, 2))
        features.append(dist(98, 327))
        
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
            features.append(dist(13, 14) / mouth_width)
        else:
            features.append(0)
        
        # Relative positions
        left_eye = [33, 160, 158, 133, 153, 144, 145, 159]
        right_eye = [362, 385, 387, 263, 373, 380, 374, 386]
        mouth = [61, 291, 13, 14]
        
        for idx in left_eye + right_eye + mouth:
            features.append(landmarks[idx][0] / (face_width + 1e-6))
            features.append(landmarks[idx][1] / (face_height + 1e-6))
        
        return np.array(features)
    
    def predict_expression(self, landmarks):
        """Predict expression from landmarks."""
        features = self.extract_features(landmarks)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # If confidence is low, return neutral
        if confidence < 0.4:
            return 'neutral', confidence
        
        return prediction, confidence
    
    def draw_ui(self, frame, expression, confidence, fps):
        """Draw clean professional UI."""
        h, w, _ = frame.shape
        
        # Clean semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Top frame border
        cv2.rectangle(frame, (0, 0), (w, 100), (255, 255, 255), 2)
        
        # Expression - clean white text
        cv2.putText(frame, f"EXPRESSION: {expression.upper()}", 
                   (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Confidence - clean white text
        cv2.putText(frame, f"CONFIDENCE: {confidence*100:.0f}%", 
                   (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Bottom info bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Bottom frame border
        cv2.rectangle(frame, (0, h-60), (w, h), (255, 255, 255), 2)
        
        # Stats - all white professional text
        stats_y = h - 20
        cv2.putText(frame, f"FPS: {fps:.0f}", 
                   (30, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MODEL ACCURACY: {self.accuracy*100:.0f}%", 
                   (w//2 - 150, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "PRESS 'Q' TO EXIT", 
                   (w - 250, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def run(self):
        """Run real-time expression detection."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Create fullscreen window
        cv2.namedWindow('ML Expression Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('ML Expression Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\n" + "="*60)
        print("ML EXPRESSION DETECTION - REAL TIME")
        print("="*60)
        print("Detecting 5 expressions: " + ", ".join(self.expressions))
        print("Low confidence → NEUTRAL")
        print("Press 'Q' to quit")
        print("="*60 + "\n")
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            expression = 'neutral'
            confidence = 0.0
            
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
                
                # Extract landmarks and predict
                landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark])
                expression, confidence = self.predict_expression(landmarks)
                
                # Smoothing
                self.prediction_history.append(expression)
                self.confidence_history.append(confidence)
                
                # Use mode for stable prediction
                if len(self.prediction_history) >= 5:
                    from collections import Counter
                    most_common = Counter(self.prediction_history).most_common(1)[0][0]
                    expression = most_common
                    confidence = np.mean(list(self.confidence_history))
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / (frame_time + 1e-6)
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)
            
            # Draw UI
            self.draw_ui(frame, expression, confidence, avg_fps)
            
            # Display
            cv2.imshow('ML Expression Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Stats
        if self.fps_history:
            print(f"\n{'='*60}")
            print(f"Performance Statistics:")
            print(f"  Average FPS: {np.mean(self.fps_history):.1f}")
            print(f"  Min FPS: {np.min(self.fps_history):.1f}")
            print(f"  Max FPS: {np.max(self.fps_history):.1f}")
            print(f"{'='*60}\n")


def main():
    try:
        detector = ExpressionMLDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run: python train_expressions_ml.py")


if __name__ == '__main__':
    main()
