import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import time

class AdvancedFaceRecognizer:
    """
    Real-time face recognition using trained face mesh models.
    Displays dense face mesh and identifies users with high accuracy.
    """
    
    def __init__(self, model_dir='trained_models'):
        self.model_dir = model_dir
        self.models = {}
        self.load_all_models()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Recognition history for smoothing
        self.recognition_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        
    def load_all_models(self):
        """Load all trained models from the model directory."""
        if not os.path.exists(self.model_dir):
            print(f"No model directory found at {self.model_dir}")
            return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_model.pkl'):
                model_path = os.path.join(self.model_dir, filename)
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        user_name = model['user_name']
                        self.models[user_name] = model
                        print(f"✓ Loaded model: {user_name} ({model['num_samples']} samples)")
                except Exception as e:
                    print(f"✗ Failed to load {filename}: {e}")
        
        if self.models:
            print(f"\nTotal models loaded: {len(self.models)}")
        else:
            print("\n⚠ No trained models found. Please train first using face_trainer.py")
    
    def _calculate_face_descriptor(self, landmarks):
        """Calculate face descriptor (same as training)."""
        descriptor = []
        
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        # Eye distances
        descriptor.append(dist(33, 133))
        descriptor.append(dist(362, 263))
        descriptor.append(dist(159, 145))
        descriptor.append(dist(386, 374))
        descriptor.append(dist(133, 263))
        
        # Nose features
        descriptor.append(dist(1, 2))
        descriptor.append(dist(98, 327))
        
        # Mouth features
        descriptor.append(dist(61, 291))
        descriptor.append(dist(0, 17))
        
        # Face proportions
        descriptor.append(dist(10, 152))
        descriptor.append(dist(234, 454))
        
        # Ratios
        face_width = dist(234, 454)
        if face_width > 0:
            descriptor.append(dist(10, 152) / face_width)
        
        return np.array(descriptor)
    
    def recognize_face(self, landmarks):
        """
        Recognize face from landmarks by comparing with trained models.
        
        Returns:
            Tuple of (user_name, confidence, similarity_score)
        """
        if not self.models:
            return "Unknown", 0.0, 0.0
        
        # Calculate descriptor for current face
        current_descriptor = self._calculate_face_descriptor(landmarks)
        
        best_match = None
        best_similarity = float('inf')
        
        # Compare with all trained models
        for user_name, model in self.models.items():
            # Calculate normalized distance
            mean_desc = model['mean_descriptor']
            std_desc = model['std_descriptor'] + 1e-6
            
            normalized_diff = (current_descriptor - mean_desc) / std_desc
            distance = np.linalg.norm(normalized_diff)
            
            if distance < best_similarity:
                best_similarity = distance
                best_match = user_name
        
        # Determine if match is valid
        if best_match and best_similarity < self.models[best_match]['recognition_threshold']:
            confidence = max(0, 1 - (best_similarity / self.models[best_match]['recognition_threshold']))
            return best_match, confidence, best_similarity
        else:
            return "Unknown", 0.0, best_similarity
    
    def draw_dense_mesh(self, frame, face_landmarks):
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
    
    def draw_ui(self, frame, user_name, confidence, fps, similarity):
        """Draw sleek UI overlay with recognition info."""
        h, w, _ = frame.shape
        
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Color coding based on recognition
        if user_name != "Unknown":
            color = (0, 255, 0)  # Green for recognized
            status = "RECOGNIZED"
        else:
            color = (0, 165, 255)  # Orange for unknown
            status = "UNKNOWN"
        
        # User name
        cv2.putText(frame, f"User: {user_name}", 
                   (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        
        # Status
        cv2.putText(frame, status, 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Confidence bar
        if confidence > 0:
            bar_width = 300
            bar_height = 25
            bar_x, bar_y = w - bar_width - 40, 30
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                        (50, 50, 50), -1)
            
            # Filled portion
            filled = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), 
                        color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                        (200, 200, 200), 2)
            
            # Text
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                       (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bottom info bar
        cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Models loaded
        cv2.putText(frame, f"Models: {len(self.models)}", 
                   (200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Similarity score
        if similarity < 1.0:
            cv2.putText(frame, f"Similarity: {similarity:.3f}", 
                       (450, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Controls
        cv2.putText(frame, "Q: Quit | T: Train New | R: Reload Models", 
                   (w - 520, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def run(self):
        """Run real-time face recognition with fullscreen display."""
        if not self.models:
            print("\n⚠ No models loaded. Please train faces first using face_trainer.py")
            response = input("\nPress Enter to continue anyway, or Ctrl+C to exit...")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Set highest resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"ADVANCED FACE RECOGNITION SYSTEM")
        print(f"{'='*60}")
        print(f"Camera Resolution: {width}x{height}")
        print(f"Loaded Models: {len(self.models)}")
        print(f"\nControls:")
        print("  Q - Quit")
        print("  T - Train new face")
        print("  R - Reload models")
        print("  F - Toggle fullscreen")
        print(f"{'='*60}\n")
        
        # Create fullscreen window
        cv2.namedWindow('Face Recognition', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        fullscreen = True
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            user_name = "No Face"
            confidence = 0.0
            similarity = 0.0
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Draw dense mesh
                self.draw_dense_mesh(frame, face_landmarks)
                
                # Extract landmarks
                landmarks = np.array([
                    [lm.x * w, lm.y * h, lm.z] 
                    for lm in face_landmarks.landmark
                ])
                
                # Recognize face
                user_name, confidence, similarity = self.recognize_face(landmarks)
                
                # Store in history for smoothing
                self.recognition_history.append(user_name)
                self.confidence_history.append(confidence)
                
                # Use mode for stable recognition
                if len(self.recognition_history) >= 5:
                    from collections import Counter
                    most_common = Counter(self.recognition_history).most_common(1)[0][0]
                    if most_common != "Unknown":
                        user_name = most_common
                        confidence = np.mean(self.confidence_history)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / (frame_time + 1e-6)
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)
            
            # Draw UI
            self.draw_ui(frame, user_name, confidence, avg_fps, similarity)
            
            # Display
            cv2.imshow('Face Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nExiting...")
                break
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord('r'):  # Reload models
                print("\nReloading models...")
                self.models.clear()
                self.load_all_models()
            elif key == ord('t'):  # Train new face
                cap.release()
                cv2.destroyAllWindows()
                
                print("\n" + "="*60)
                from face_trainer import FaceMeshTrainer
                trainer = FaceMeshTrainer()
                
                user_name_input = input("Enter name for new face: ").strip()
                if user_name_input:
                    trainer.capture_training_samples(user_name_input, 200)
                    self.load_all_models()
                
                # Restart camera
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                cv2.namedWindow('Face Recognition', cv2.WND_PROP_FULLSCREEN)
                if fullscreen:
                    cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if self.fps_history:
            print(f"\nPerformance Statistics:")
            print(f"  Average FPS: {np.mean(self.fps_history):.1f}")
            print(f"  Min FPS: {np.min(self.fps_history):.1f}")
            print(f"  Max FPS: {np.max(self.fps_history):.1f}")


def main():
    recognizer = AdvancedFaceRecognizer()
    recognizer.run()


if __name__ == '__main__':
    main()
