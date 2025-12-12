"""
DTW Template Prediction - Test against new gesture data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import json

def predict_sign(new_sequence, template, threshold, features):
    """
    Predict if a new gesture matches the template
    
    Args:
        new_sequence: DataFrame or array with MPU data
        template: Trained template
        threshold: Detection threshold
        features: List of feature column names
    
    Returns:
        dict with prediction, confidence, distance
    """
    # Extract MPU features
    if isinstance(new_sequence, pd.DataFrame):
        sequence_data = new_sequence[features].values
    else:
        sequence_data = new_sequence
    
    # Normalize (z-score)
    normalized = stats.zscore(sequence_data, axis=0, nan_policy='omit')
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    # Calculate DTW distance using fastdtw
    distance, _ = fastdtw(normalized, template, dist=euclidean)
    
    # Calculate confidence
    confidence = 1 / (1 + distance)
    
    # Make prediction
    prediction = "J" if distance < threshold else "NOT J"
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'distance': distance,
        'threshold': threshold,
        'is_match': distance < threshold
    }

print("="*70)
print(" DTW TEMPLATE PREDICTION - SIGN LANGUAGE RECOGNITION")
print("="*70)

# Load template and config
print("\n📁 Loading trained template...")
template = np.load('template_J.npy')

with open('template_J_config.json', 'r') as f:
    config = json.load(f)

print(f"   ✅ Template loaded for sign: {config['sign']}")
print(f"   Threshold: {config['threshold']:.4f}")
print(f"   Template shape: {template.shape}")

# Test on a sample from training data
print("\n🧪 Testing prediction function...")
df = pd.read_csv('J_dual_dataset_20251212_063557.csv')

# Test on trial 1
print("\n   Testing on Trial 1 (should be recognized as 'J'):")
trial_1 = df[df['trial_id'] == 1]

result = predict_sign(trial_1, template, config['threshold'], config['features'])

print(f"   Prediction: {result['prediction']}")
print(f"   Confidence: {result['confidence']:.2%}")
print(f"   Distance: {result['distance']:.4f}")
print(f"   Threshold: {result['threshold']:.4f}")
print(f"   Match: {'✓ YES' if result['is_match'] else '✗ NO'}")

# Test on last trial
last_trial_id = df['trial_id'].max()
print(f"\n   Testing on Trial {last_trial_id} (should be recognized as 'J'):")
trial_last = df[df['trial_id'] == last_trial_id]

result = predict_sign(trial_last, template, config['threshold'], config['features'])

print(f"   Prediction: {result['prediction']}")
print(f"   Confidence: {result['confidence']:.2%}")
print(f"   Distance: {result['distance']:.4f}")
print(f"   Match: {'✓ YES' if result['is_match'] else '✗ NO'}")

print("\n" + "="*70)
print("✅ PREDICTION SYSTEM READY!")
print("="*70)
print("\nTo add more signs:")
print("  1. Collect data for new sign (e.g., 'K', 'L', etc.)")
print("  2. Run train_dtw_template.py with new dataset")
print("  3. Create template_K.npy, template_L.npy, etc.")
print("\nFor multi-sign recognition:")
print("  - Load all templates")
print("  - Compute distance to each")
print("  - Pick minimum distance")
print("  - Check if below that sign's threshold")
