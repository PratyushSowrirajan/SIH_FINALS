import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import json

class FaceMeshTrainer:
    """
    Advanced face mesh trainer that captures face samples and creates a personalized model.
    Uses dense mesh representation for high accuracy recognition.
    """
    
    def __init__(self, samples_dir='face_samples', model_dir='trained_models'):
        self.samples_dir = samples_dir
        self.model_dir = model_dir
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize MediaPipe Face Mesh with maximum accuracy
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
        
        # Face mesh drawing specification - clean white mesh only
        self.tesselation_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=0,
            color=(255, 255, 255)
        )
        
    def capture_training_samples(self, user_name, num_frames=200):
        """
        Capture face samples for training.
        
        Args:
            user_name: Name identifier for the user
            num_frames: Number of frames to capture (default: 200)
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Set camera to highest resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"FACE MESH TRAINING - Capture Mode")
        print(f"{'='*60}")
        print(f"User: {user_name}")
        print(f"Target Frames: {num_frames}")
        print(f"Camera Resolution: {width}x{height}")
        print(f"\nInstructions:")
        print("  - Position your face in the center")
        print("  - Move your head slowly (left, right, up, down)")
        print("  - Change expressions naturally")
        print("  - Maintain good lighting")
        print("  - Press SPACE to start capturing")
        print("  - Press ESC to cancel")
        print(f"{'='*60}\n")
        
        samples = []
        frame_count = 0
        capturing = False
        
        # Create fullscreen window
        cv2.namedWindow('Face Mesh Training', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Face Mesh Training', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            # Create overlay
            overlay = frame.copy()
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Draw clean white mesh network only
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.tesselation_spec
                )
                
                # Capture samples
                if capturing and frame_count < num_frames:
                    landmarks_array = np.array([
                        [lm.x, lm.y, lm.z] 
                        for lm in face_landmarks.landmark
                    ])
                    
                    samples.append({
                        'landmarks': landmarks_array,
                        'timestamp': datetime.now().isoformat(),
                        'frame_number': frame_count
                    })
                    
                    frame_count += 1
                    
                    # Visual feedback
                    cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
            
            # Draw UI
            if not capturing:
                # Instructions overlay
                cv2.rectangle(frame, (0, 0), (w, 150), (0, 0, 0), -1)
                cv2.putText(frame, "Position your face and press SPACE to start", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(frame, "ESC to cancel", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            else:
                # Progress bar
                progress = frame_count / num_frames
                bar_width = w - 100
                bar_height = 40
                bar_x, bar_y = 50, h - 100
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                            (50, 50, 50), -1)
                
                # Progress
                filled_width = int(bar_width * progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                            (0, 255, 0), -1)
                
                # Text
                cv2.putText(frame, f"Capturing: {frame_count}/{num_frames} ({progress*100:.1f}%)", 
                           (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(frame, "Slowly move your head in different angles", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Face Mesh Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\nCapture cancelled by user.")
                break
            elif key == 32 and not capturing:  # SPACE
                capturing = True
                print("\nStarting capture...")
            
            if frame_count >= num_frames:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(samples) >= num_frames * 0.8:  # At least 80% captured
            # Save samples
            user_dir = os.path.join(self.samples_dir, user_name)
            os.makedirs(user_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(user_dir, f"samples_{timestamp}.pkl")
            
            with open(filename, 'wb') as f:
                pickle.dump({
                    'user_name': user_name,
                    'samples': samples,
                    'num_frames': len(samples),
                    'timestamp': timestamp
                }, f)
            
            print(f"\n✓ Captured {len(samples)} samples successfully!")
            print(f"✓ Saved to: {filename}")
            
            # Train model immediately
            print("\nTraining model...")
            self.train_model(user_name)
            
            return True
        else:
            print(f"\n✗ Insufficient samples captured: {len(samples)}/{num_frames}")
            return False
    
    def train_model(self, user_name):
        """
        Train a recognition model from captured samples.
        
        Args:
            user_name: Name identifier for the user
        """
        user_dir = os.path.join(self.samples_dir, user_name)
        
        # Load all sample files for this user
        all_samples = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.pkl'):
                with open(os.path.join(user_dir, filename), 'rb') as f:
                    data = pickle.load(f)
                    all_samples.extend(data['samples'])
        
        if len(all_samples) < 50:
            print(f"✗ Not enough samples to train (need 50+, got {len(all_samples)})")
            return False
        
        print(f"\nTraining with {len(all_samples)} samples...")
        
        # Extract landmark features
        landmarks_list = [sample['landmarks'] for sample in all_samples]
        landmarks_array = np.array(landmarks_list)
        
        # Calculate statistics for the user's face
        mean_landmarks = np.mean(landmarks_array, axis=0)
        std_landmarks = np.std(landmarks_array, axis=0)
        
        # Calculate face shape descriptors
        face_descriptors = []
        for landmarks in landmarks_array:
            descriptor = self._calculate_face_descriptor(landmarks)
            face_descriptors.append(descriptor)
        
        face_descriptors = np.array(face_descriptors)
        mean_descriptor = np.mean(face_descriptors, axis=0)
        std_descriptor = np.std(face_descriptors, axis=0)
        
        # Create model
        model = {
            'user_name': user_name,
            'mean_landmarks': mean_landmarks,
            'std_landmarks': std_landmarks,
            'mean_descriptor': mean_descriptor,
            'std_descriptor': std_descriptor,
            'num_samples': len(all_samples),
            'trained_date': datetime.now().isoformat(),
            'recognition_threshold': 0.15  # Similarity threshold
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{user_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Model trained successfully!")
        print(f"✓ Saved to: {model_path}")
        print(f"✓ Samples used: {len(all_samples)}")
        
        return True
    
    def _calculate_face_descriptor(self, landmarks):
        """
        Calculate a compact face descriptor from landmarks.
        Uses geometric features and distances between key points.
        """
        descriptor = []
        
        # Key facial regions
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        nose_indices = [1, 2, 98, 327]
        mouth_indices = [61, 291, 0, 17, 269, 267]
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Calculate distances between key points
        def dist(i1, i2):
            return np.linalg.norm(landmarks[i1] - landmarks[i2])
        
        # Eye distances
        descriptor.append(dist(33, 133))  # Left eye width
        descriptor.append(dist(362, 263))  # Right eye width
        descriptor.append(dist(159, 145))  # Left eye height
        descriptor.append(dist(386, 374))  # Right eye height
        descriptor.append(dist(133, 263))  # Inter-eye distance
        
        # Nose features
        descriptor.append(dist(1, 2))  # Nose bridge
        descriptor.append(dist(98, 327))  # Nose width
        
        # Mouth features
        descriptor.append(dist(61, 291))  # Mouth width
        descriptor.append(dist(0, 17))  # Mouth height
        
        # Face proportions
        descriptor.append(dist(10, 152))  # Face height
        descriptor.append(dist(234, 454))  # Face width
        
        # Ratios
        face_width = dist(234, 454)
        if face_width > 0:
            descriptor.append(dist(10, 152) / face_width)  # Face aspect ratio
        
        return np.array(descriptor)


def main():
    trainer = FaceMeshTrainer()
    
    print("\n" + "="*60)
    print("ADVANCED FACE MESH TRAINING SYSTEM")
    print("="*60)
    
    user_name = input("\nEnter your name: ").strip()
    
    if not user_name:
        print("Invalid name. Exiting.")
        return
    
    num_frames = 200
    response = input(f"\nCapture {num_frames} frames? (Y/n): ").strip().lower()
    
    if response and response != 'y':
        try:
            num_frames = int(input("Enter number of frames: "))
        except:
            print("Invalid input. Using default 200 frames.")
            num_frames = 200
    
    # Start training
    trainer.capture_training_samples(user_name, num_frames)


if __name__ == '__main__':
    main()
