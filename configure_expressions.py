import cv2
import json
import numpy as np
from pathlib import Path
from expression_detector import ExpressionDetector
import mediapipe as mp
from typing import Dict, List

class ExpressionConfigurator:
    """
    Interactive tool to calibrate and create custom facial expressions.
    Collects expression samples and generates optimal thresholds.
    """
    
    def __init__(self, config_file: str = 'expressions.json'):
        self.config_file = config_file
        self.expressions_data = self._load_config()
        self.detector = ExpressionDetector()
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
        )
        
    def _load_config(self) -> Dict:
        """Load existing configuration."""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.expressions_data, f, indent=2)
        print(f"Configuration saved to {self.config_file}")
    
    def calibrate_expression(self, expression_name: str, num_samples: int = 15):
        """
        Calibrate a custom expression by collecting samples.
        
        Args:
            expression_name: Name of the expression to calibrate
            num_samples: Number of samples to collect
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n=== Calibrating Expression: {expression_name.upper()} ===")
        print(f"Collect {num_samples} samples of this expression.")
        print("Press SPACE to capture a sample, 'q' to finish, 'r' to reset.\n")
        
        samples = []
        sample_count = 0
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Detect face mesh
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            # Draw status
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            cv2.putText(frame, f"Expression: {expression_name.upper()}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow('Expression Calibration', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and results.multi_face_landmarks:
                # Extract landmarks
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([
                    [lm.x * w, lm.y * h, lm.z] 
                    for lm in face_landmarks.landmark
                ])
                
                # Extract features
                features = self.detector.extract_features(landmarks)
                samples.append(features)
                sample_count += 1
                print(f"Sample {sample_count} captured")
                
            elif key == ord('q'):
                break
            elif key == ord('r'):
                samples = []
                sample_count = 0
                print("Reset!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if sample_count >= 3:
            thresholds = self._compute_thresholds(samples)
            self.expressions_data[expression_name] = thresholds
            self._save_config()
            print(f"\nExpression '{expression_name}' calibrated successfully!")
            print(f"Thresholds: {json.dumps(thresholds, indent=2)}")
            return True
        else:
            print(f"Not enough samples. Need at least 3, got {sample_count}.")
            return False
    
    def _compute_thresholds(self, samples: List[Dict]) -> Dict:
        """Compute optimal thresholds from collected samples."""
        thresholds = {}
        
        feature_names = set()
        for sample in samples:
            feature_names.update(sample.keys())
        
        for feature in feature_names:
            values = [sample[feature] for sample in samples if feature in sample]
            
            if values:
                min_val = min(values)
                max_val = max(values)
                margin = (max_val - min_val) * 0.2
                
                thresholds[feature] = [
                    max(0, min_val - margin),
                    max_val + margin
                ]
        
        return thresholds
    
    def list_expressions(self):
        """List all configured expressions."""
        print("\n=== Configured Expressions ===")
        if self.expressions_data:
            for i, expr_name in enumerate(self.expressions_data.keys(), 1):
                print(f"{i}. {expr_name}")
        else:
            print("No expressions configured yet.")
    
    def delete_expression(self, expression_name: str):
        """Delete a configured expression."""
        if expression_name in self.expressions_data:
            del self.expressions_data[expression_name]
            self._save_config()
            print(f"Expression '{expression_name}' deleted.")
        else:
            print(f"Expression '{expression_name}' not found.")
    
    def interactive_menu(self):
        """Interactive configuration menu."""
        while True:
            print("\n=== Facial Expression Calibrator ===")
            print("1. Calibrate new expression")
            print("2. List expressions")
            print("3. Delete expression")
            print("4. Test expression detection")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                expr_name = input("Enter expression name: ").strip().lower()
                if expr_name:
                    self.calibrate_expression(expr_name)
                    
            elif choice == '2':
                self.list_expressions()
                
            elif choice == '3':
                self.list_expressions()
                expr_name = input("Enter expression name to delete: ").strip().lower()
                self.delete_expression(expr_name)
                
            elif choice == '4':
                self.test_expressions()
                
            elif choice == '5':
                break


def main():
    configurator = ExpressionConfigurator('expressions.json')
    configurator.interactive_menu()


if __name__ == '__main__':
    main()
