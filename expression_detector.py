import numpy as np
from typing import Dict, Tuple, List

class ExpressionDetector:
    """
    Detects facial expressions using MediaPipe Face Mesh landmarks.
    Calculates features from landmark positions to classify expressions.
    """
    
    def __init__(self):
        # Key landmark indices for expression detection
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 95, 88, 178, 87, 10, 152, 155, 133, 246, 161, 160, 159, 158, 157, 173]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]
        self.NOSE = [1, 2, 4, 5, 6, 8, 9, 10, 19, 94, 98, 195, 197, 198, 209, 210, 218, 220, 399, 456]
        
    def get_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)
    
    def get_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p2 is vertex)."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    
    def extract_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract relevant features from face mesh landmarks."""
        features = {}
        
        # Eye openness (distance between upper and lower eyelid)
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        left_eye_openness = self.get_distance(left_eye_top, left_eye_bottom)
        
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        right_eye_openness = self.get_distance(right_eye_top, right_eye_bottom)
        
        features['left_eye_openness'] = left_eye_openness
        features['right_eye_openness'] = right_eye_openness
        features['avg_eye_openness'] = (left_eye_openness + right_eye_openness) / 2
        
        # Mouth openness (vertical distance)
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_openness = self.get_distance(mouth_top, mouth_bottom)
        
        # Mouth width
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_width = self.get_distance(mouth_left, mouth_right)
        
        features['mouth_openness'] = mouth_openness
        features['mouth_width'] = mouth_width
        features['mouth_aspect_ratio'] = mouth_openness / (mouth_width + 1e-6)
        
        # Eyebrow position (vertical distance from nose)
        nose_top = landmarks[10]
        left_eyebrow = landmarks[105]
        right_eyebrow = landmarks[334]
        
        features['left_eyebrow_raise'] = abs(landmarks[10][1] - left_eyebrow[1])
        features['right_eyebrow_raise'] = abs(landmarks[10][1] - right_eyebrow[1])
        features['avg_eyebrow_raise'] = (features['left_eyebrow_raise'] + features['right_eyebrow_raise']) / 2
        
        # Nostril flare (distance between nostrils)
        left_nostril = landmarks[331]
        right_nostril = landmarks[99]
        features['nostril_flare'] = self.get_distance(left_nostril, right_nostril)
        
        # Face tilt (using chin and forehead)
        chin = landmarks[152]
        forehead = landmarks[10]
        features['face_tilt'] = abs(chin[0] - forehead[0])
        
        # Lip corner elevation
        lip_left = landmarks[61]
        lip_right = landmarks[291]
        mouth_center = landmarks[0]
        
        features['lip_corner_left_elevation'] = mouth_center[1] - lip_left[1]
        features['lip_corner_right_elevation'] = mouth_center[1] - lip_right[1]
        
        return features
    
    def classify_expression(self, landmarks: np.ndarray, custom_expressions: Dict = None) -> Tuple[str, float]:
        """
        Classify facial expression based on extracted features.
        
        Args:
            landmarks: Face mesh landmarks (468 x 3 array)
            custom_expressions: Optional dictionary of custom expression definitions
            
        Returns:
            Tuple of (expression_name, confidence)
        """
        features = self.extract_features(landmarks)
        
        expressions = custom_expressions if custom_expressions else self._get_default_expressions()
        
        best_match = 'neutral'
        best_score = -float('inf')
        
        for expr_name, expr_thresholds in expressions.items():
            score = self._calculate_match_score(features, expr_thresholds)
            if score > best_score:
                best_score = score
                best_match = expr_name
        
        # Confidence score (0-1)
        confidence = max(0, min(1, (best_score + 10) / 20))
        
        return best_match, confidence
    
    def _get_default_expressions(self) -> Dict:
        """Default expression definitions."""
        return {
            'neutral': {
                'mouth_openness': (0.01, 0.15),
                'avg_eye_openness': (0.08, 0.25),
                'avg_eyebrow_raise': (0.01, 0.15),
            },
            'happy': {
                'mouth_openness': (0.08, 0.50),
                'mouth_width': (0.3, 1.0),
                'avg_eye_openness': (0.05, 0.30),
                'lip_corner_left_elevation': (0.05, float('inf')),
                'lip_corner_right_elevation': (0.05, float('inf')),
            },
            'sad': {
                'mouth_openness': (0.01, 0.15),
                'mouth_aspect_ratio': (0.01, 0.08),
                'avg_eyebrow_raise': (0.05, float('inf')),
                'lip_corner_left_elevation': (-float('inf'), -0.01),
                'lip_corner_right_elevation': (-float('inf'), -0.01),
            },
            'surprised': {
                'mouth_openness': (0.15, 0.60),
                'avg_eye_openness': (0.20, float('inf')),
                'avg_eyebrow_raise': (0.15, float('inf')),
            },
            'angry': {
                'mouth_openness': (0.01, 0.20),
                'avg_eye_openness': (0.05, 0.20),
                'avg_eyebrow_raise': (-0.10, 0.01),
                'nostril_flare': (0.08, float('inf')),
            },
            'disgusted': {
                'mouth_openness': (0.05, 0.25),
                'mouth_aspect_ratio': (0.05, 0.15),
                'nostril_flare': (0.07, float('inf')),
                'avg_eye_openness': (0.05, 0.18),
            },
        }
    
    def _calculate_match_score(self, features: Dict, thresholds: Dict) -> float:
        """Calculate how well features match expression thresholds."""
        score = 0
        matched = 0
        
        for feature_name, (min_val, max_val) in thresholds.items():
            if feature_name not in features:
                continue
            
            feature_val = features[feature_name]
            
            # Check if feature is within threshold
            if min_val <= feature_val <= max_val:
                score += 1
                matched += 1
            elif feature_val < min_val:
                score -= abs(feature_val - min_val) * 2
            else:
                score -= abs(feature_val - max_val) * 2
        
        return score / max(matched, 1) if matched > 0 else -10
