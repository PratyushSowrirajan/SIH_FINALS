import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from expression_detector import ExpressionDetector
from typing import Dict, Optional
import threading
import queue
from collections import deque

class FacialExpressionRecognizer:
    """
    Real-time facial expression recognition system using MediaPipe Face Mesh.
    Optimized for low latency and high performance.
    """
    
    def __init__(self, config_file: str = 'expressions.json'):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = ExpressionDetector()
        self.config_file = config_file
        self.custom_expressions = self._load_expressions()
        
        # Performance optimization
        self.frame_buffer = deque(maxlen=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.expression_history = deque(maxlen=5)
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'neutral': (200, 200, 200),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprised': (255, 165, 0),
            'angry': (0, 0, 255),
            'disgusted': (128, 0, 128),
        }
        
    def _load_expressions(self) -> Optional[Dict]:
        """Load custom expression definitions from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading expressions: {e}")
        return None
    
    def save_expression(self, name: str, thresholds: Dict) -> bool:
        """Save a custom expression definition."""
        try:
            expressions = self.custom_expressions or {}
            expressions[name] = thresholds
            
            with open(self.config_file, 'w') as f:
                json.dump(expressions, f, indent=2)
            
            self.custom_expressions = expressions
            return True
        except Exception as e:
            print(f"Error saving expression: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame to detect faces and expressions.
        Returns (processed_frame, expression, confidence)
        """
        h, w, c = frame.shape
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face mesh
        results = self.face_mesh.process(frame_rgb)
        
        expression = 'neutral'
        confidence = 0.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array
            landmarks = np.array([
                [lm.x * w, lm.y * h, lm.z] 
                for lm in face_landmarks.landmark
            ])
            
            # Classify expression
            expression, confidence = self.detector.classify_expression(
                landmarks, 
                self.custom_expressions
            )
            
            # Store in history for smoothing
            self.expression_history.append(expression)
            
            # Draw clean white mesh network
            self._draw_face_mesh(frame, face_landmarks, h, w)
        
        return frame, expression, confidence
    
    def _draw_face_mesh(self, frame: np.ndarray, face_landmarks, h: int, w: int):
        """Draw clean white mesh network only."""
        # White mesh specification
        white_mesh_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=0,
            color=(255, 255, 255)
        )
        
        # Draw only tesselation (full mesh network) in white
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=white_mesh_spec
        )
    
    def _draw_landmarks_highlights(self, frame: np.ndarray, landmarks: np.ndarray):
        """Highlight important facial features."""
        # No highlights - clean mesh only
        pass
    
    def draw_ui(self, frame: np.ndarray, expression: str, confidence: float) -> np.ndarray:
        """Draw sleek UI with expression info."""
        h, w, _ = frame.shape
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Expression text
        color = self.colors.get(expression, (200, 200, 200))
        cv2.putText(
            frame,
            f"Expression: {expression.upper()}",
            (20, 40),
            self.font,
            1.2,
            color,
            2
        )
        
        # Confidence bar
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 20, 60
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Filled portion
        filled_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), color, -1)
        
        # Text
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (bar_x + 50, bar_y + 35),
            self.font,
            0.6,
            (255, 255, 255),
            1
        )
        
        # FPS indicator
        cv2.putText(
            frame,
            "Press 'q' to quit | 'c' to configure",
            (10, h - 10),
            self.font,
            0.5,
            (200, 200, 200),
            1
        )
        
        return frame
    
    def run(self, camera_id: int = 0, target_fps: int = 30):
        """Run real-time facial expression recognition."""
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        # Set resolution for fullscreen
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        frame_time = 1 / target_fps
        
        print("Starting Facial Expression Recognition System...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Configure custom expression")
        print("  's' - Save current frame")
        print("  't' - Train face recognition (200 frames)")
        print("")
        
        # Create fullscreen window
        cv2.namedWindow('Facial Expression Recognition', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Facial Expression Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        smoothed_expression = 'neutral'
        smoothed_confidence = 0.0
        training_mode = False
        training_frames = []
        training_target = 200
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Training mode - collect frames
            if training_mode and len(training_frames) < training_target:
                # Process for mesh display
                frame_processed, expression, confidence = self.process_frame(frame)
                
                # Store frame data
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark])
                    training_frames.append({'landmarks': landmarks})
                
                # Draw training progress
                progress = len(training_frames) / training_target
                bar_width = w - 100
                bar_height = 50
                bar_x, bar_y = 50, h // 2 - 25
                
                overlay = frame_processed.copy()
                cv2.rectangle(overlay, (0, bar_y - 80), (w, bar_y + bar_height + 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame_processed, 0.5, 0, frame_processed)
                
                cv2.putText(frame_processed, "TRAINING MODE - Collecting Face Data", 
                           (w//2 - 350, bar_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Progress bar
                cv2.rectangle(frame_processed, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                filled = int(bar_width * progress)
                cv2.rectangle(frame_processed, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 255, 0), -1)
                cv2.putText(frame_processed, f"{len(training_frames)}/{training_target} frames ({progress*100:.1f}%)", 
                           (bar_x + 20, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                frame = frame_processed
                
                # Check if training complete
                if len(training_frames) >= training_target:
                    print("\n✓ Training data collected! Processing...")
                    self._train_face_model(training_frames)
                    training_mode = False
                    training_frames = []
            else:
                # Process frame for expression detection
                frame, expression, confidence = self.process_frame(frame)
                
                # Smooth expression transitions
                if self.expression_history:
                    smoothed_expression = max(self.expression_history, 
                                             key=self.expression_history.count)
                    smoothed_confidence = confidence * 0.7 + smoothed_confidence * 0.3
                
                # Draw UI
                frame = self.draw_ui(frame, smoothed_expression, confidence)
            
            # Display
            cv2.imshow('Facial Expression Recognition', frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('c'):
                self._configure_expression()
            elif key == ord('s'):
                filename = f"screenshot_{len(os.listdir('.') or [])}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('t') and not training_mode:
                print("\n" + "="*60)
                print("Starting Face Recognition Training")
                print("="*60)
                user_name = input("Enter your name: ").strip()
                if user_name:
                    print(f"Collecting 200 frames for {user_name}...")
                    print("Move your head slowly in different angles.")
                    training_mode = True
                    training_frames = []
                    self.current_training_user = user_name
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _train_face_model(self, training_frames):
        """Train face recognition model from collected frames."""
        import pickle
        from datetime import datetime
        
        os.makedirs('face_samples', exist_ok=True)
        os.makedirs('trained_models', exist_ok=True)
        
        # Extract landmarks
        landmarks_list = [frame['landmarks'] for frame in training_frames]
        landmarks_array = np.array(landmarks_list)
        
        # Calculate statistics
        mean_landmarks = np.mean(landmarks_array, axis=0)
        std_landmarks = np.std(landmarks_array, axis=0)
        
        # Calculate face descriptors
        face_descriptors = []
        for landmarks in landmarks_array:
            descriptor = self._calculate_face_descriptor(landmarks)
            face_descriptors.append(descriptor)
        
        face_descriptors = np.array(face_descriptors)
        mean_descriptor = np.mean(face_descriptors, axis=0)
        std_descriptor = np.std(face_descriptors, axis=0)
        
        # Create model
        model = {
            'user_name': self.current_training_user,
            'mean_landmarks': mean_landmarks,
            'std_landmarks': std_landmarks,
            'mean_descriptor': mean_descriptor,
            'std_descriptor': std_descriptor,
            'num_samples': len(training_frames),
            'trained_date': datetime.now().isoformat(),
            'recognition_threshold': 0.15
        }
        
        # Save model
        model_path = os.path.join('trained_models', f"{self.current_training_user}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save samples
        samples_path = os.path.join('face_samples', f"{self.current_training_user}_samples.pkl")
        with open(samples_path, 'wb') as f:
            pickle.dump(training_frames, f)
        
        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"{'='*60}")
        print(f"User: {self.current_training_user}")
        print(f"Samples: {len(training_frames)}")
        print(f"Model saved: {model_path}")
        print(f"Samples saved: {samples_path}")
        print(f"{'='*60}\n")
    
    def _calculate_face_descriptor(self, landmarks):
        """Calculate face descriptor from landmarks."""
        descriptor = []
        
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        # Key distances
        descriptor.append(dist(33, 133))  # Left eye
        descriptor.append(dist(362, 263))  # Right eye
        descriptor.append(dist(159, 145))  # Left eye height
        descriptor.append(dist(386, 374))  # Right eye height
        descriptor.append(dist(133, 263))  # Inter-eye
        descriptor.append(dist(1, 2))  # Nose bridge
        descriptor.append(dist(98, 327))  # Nose width
        descriptor.append(dist(61, 291))  # Mouth width
        descriptor.append(dist(0, 17))  # Mouth height
        descriptor.append(dist(10, 152))  # Face height
        descriptor.append(dist(234, 454))  # Face width
        
        face_width = dist(234, 454)
        if face_width > 0:
            descriptor.append(dist(10, 152) / face_width)
        
        return np.array(descriptor)
    
    def _configure_expression(self):
        """Interactive expression configuration."""
        print("\n=== Expression Configuration ===")
        expr_name = input("Enter expression name: ").lower().strip()
        
        if not expr_name:
            print("Cancelled.")
            return
        
        print(f"\nConfiguring '{expr_name}'...")
        print("Leave blank to skip parameter.")
        
        thresholds = {}
        
        params = {
            'mouth_openness': 'Mouth opening (0.0-0.5)',
            'mouth_width': 'Mouth width (0.0-1.0)',
            'mouth_aspect_ratio': 'Mouth aspect ratio (0.0-0.2)',
            'avg_eye_openness': 'Average eye openness (0.0-0.3)',
            'avg_eyebrow_raise': 'Average eyebrow raise (0.0-0.3)',
            'nostril_flare': 'Nostril flare (0.0-0.2)',
            'lip_corner_elevation': 'Lip corner elevation (-0.2 to 0.2)',
        }
        
        for param, description in params.items():
            try:
                min_val = input(f"  {description} - Min value: ")
                if not min_val.strip():
                    continue
                
                max_val = input(f"  {description} - Max value: ")
                if not max_val.strip():
                    continue
                
                thresholds[param] = (float(min_val), float(max_val))
            except ValueError:
                print(f"  Invalid input for {param}, skipping...")
        
        if thresholds:
            self.save_expression(expr_name, thresholds)
            print(f"\nExpression '{expr_name}' saved successfully!")
        else:
            print("No parameters configured.")
        
        print("Returning to camera feed...\n")


def main():
    recognizer = FacialExpressionRecognizer('expressions.json')
    recognizer.run(camera_id=0, target_fps=30)


if __name__ == '__main__':
    main()
