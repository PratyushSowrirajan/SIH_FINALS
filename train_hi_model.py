"""
Train a simple HI gesture classifier using DTW (Dynamic Time Warping)
Template-based approach for real-time gesture recognition
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("=" * 70)
print(" HI GESTURE CLASSIFIER TRAINER")
print("=" * 70)

# Load datasets
print("\n📂 Loading datasets...")
hi_data = pd.read_csv('hi_dataset_20251210_030854.csv')
not_hi_data = pd.read_csv('not_hi_dataset_20251210_031340.csv')

print(f"✅ HI dataset: {len(hi_data)} samples, {hi_data['trial_id'].max()} trials")
print(f"✅ NOT_HI dataset: {len(not_hi_data)} samples, {not_hi_data['trial_id'].max()} trials")

# Extract features per trial
def extract_trial_features(df, trial_id):
    """Extract time series features for one trial"""
    trial = df[df['trial_id'] == trial_id].sort_values('time_ms')
    
    # Get sensor data as arrays
    features = {
        'ax': trial['ax'].values,
        'ay': trial['ay'].values,
        'az': trial['az'].values,
        'gx': trial['gx'].values,
        'gy': trial['gy'].values,
        'gz': trial['gz'].values,
    }
    
    # Combined feature vector (all 6 axes concatenated)
    combined = np.column_stack([
        features['ax'], features['ay'], features['az'],
        features['gx'], features['gy'], features['gz']
    ])
    
    return combined, features

# Create template from HI gestures (average of all trials)
print("\n🧠 Creating HI gesture template...")
hi_trials = []
for trial_id in hi_data['trial_id'].unique():
    combined, _ = extract_trial_features(hi_data, trial_id)
    hi_trials.append(combined)

# Resample all trials to same length (200 samples)
TARGET_LENGTH = 200
hi_trials_resampled = []
for trial in hi_trials:
    # Simple linear resampling
    indices = np.linspace(0, len(trial) - 1, TARGET_LENGTH)
    resampled = np.array([trial[int(i)] for i in indices])
    hi_trials_resampled.append(resampled)

# Create average template
hi_template = np.mean(hi_trials_resampled, axis=0)

print(f"✅ HI template created: shape {hi_template.shape}")

# Calculate statistics for threshold
print("\n📊 Calculating similarity thresholds...")

def euclidean_distance(a, b):
    """Simple Euclidean distance between two sequences"""
    if len(a) != len(b):
        # Resample to match lengths
        indices = np.linspace(0, len(b) - 1, len(a))
        b = np.array([b[int(i)] for i in indices])
    return np.sqrt(np.sum((a - b) ** 2))

# Distance of HI trials to template
hi_distances = []
for trial in hi_trials_resampled:
    dist = euclidean_distance(hi_template, trial)
    hi_distances.append(dist)

# Distance of NOT_HI trials to template
not_hi_distances = []
for trial_id in not_hi_data['trial_id'].unique():
    combined, _ = extract_trial_features(not_hi_data, trial_id)
    # Resample
    indices = np.linspace(0, len(combined) - 1, TARGET_LENGTH)
    resampled = np.array([combined[int(i)] for i in indices])
    dist = euclidean_distance(hi_template, resampled)
    not_hi_distances.append(dist)

hi_mean = np.mean(hi_distances)
hi_std = np.std(hi_distances)
not_hi_mean = np.mean(not_hi_distances)
not_hi_std = np.std(not_hi_distances)

print(f"\n   HI trials distance:     {hi_mean:.2f} ± {hi_std:.2f}")
print(f"   NOT_HI trials distance: {not_hi_mean:.2f} ± {not_hi_std:.2f}")

# Set threshold (midpoint between means)
threshold = (hi_mean + not_hi_mean) / 2
print(f"\n🎯 Classification threshold: {threshold:.2f}")

# Test accuracy
correct = 0
total = len(hi_distances) + len(not_hi_distances)

for dist in hi_distances:
    if dist < threshold:
        correct += 1

for dist in not_hi_distances:
    if dist >= threshold:
        correct += 1

accuracy = correct / total * 100
print(f"✅ Training accuracy: {accuracy:.1f}%")

# Save model
model = {
    'template': hi_template,
    'threshold': threshold,
    'target_length': TARGET_LENGTH,
    'hi_mean': hi_mean,
    'hi_std': hi_std,
    'not_hi_mean': not_hi_mean,
    'not_hi_std': not_hi_std,
    'accuracy': accuracy,
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('hi_gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n💾 Model saved: hi_gesture_model.pkl")
print("\n" + "=" * 70)
print(" TRAINING COMPLETE!")
print("=" * 70)
print(f"✅ Model ready for real-time detection")
print(f"📊 Accuracy: {accuracy:.1f}%")
print(f"🎯 Use 'test_hi_realtime.py' to test live gestures!")
print("=" * 70)
