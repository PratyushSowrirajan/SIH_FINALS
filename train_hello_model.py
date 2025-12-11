"""
Train a 'hello' sign classifier
Uses template matching with all 11 features (flex1-5 + ax,ay,az + gx,gy,gz)
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("=" * 70)
print(" 'HELLO' SIGN CLASSIFIER TRAINER")
print("=" * 70)

# Load dataset
print("\n📂 Loading dataset...")
data = pd.read_csv('hello_dataset_20251210_112916.csv')

print(f"✅ Dataset loaded: {len(data)} samples, {data['trial_id'].max()} trials")
print(f"   Sign: {data['sign_name'].iloc[0]}")

# Extract features per trial
def extract_trial_features(df, trial_id):
    """Extract time series features for one trial"""
    trial = df[df['trial_id'] == trial_id].sort_values('time_ms')
    
    # Get all 11 sensor features
    features = np.column_stack([
        trial['flex1'].values,
        trial['flex2'].values,
        trial['flex3'].values,
        trial['flex4'].values,
        trial['flex5'].values,
        trial['ax'].values,
        trial['ay'].values,
        trial['az'].values,
        trial['gx'].values,
        trial['gy'].values,
        trial['gz'].values
    ])
    
    return features

# Create template from all trials
print("\n🧠 Creating 'hello' sign template...")
trials = []
trial_ids = data['trial_id'].unique()

for trial_id in trial_ids:
    features = extract_trial_features(data, trial_id)
    trials.append(features)

# Resample all trials to same length (300 samples @ ~100Hz for 3 seconds)
TARGET_LENGTH = 300
trials_resampled = []
for trial in trials:
    # Simple linear resampling
    indices = np.linspace(0, len(trial) - 1, TARGET_LENGTH)
    resampled = np.array([trial[int(i)] for i in indices])
    trials_resampled.append(resampled)

# Create average template
template = np.mean(trials_resampled, axis=0)

print(f"✅ Template created: shape {template.shape}")
print(f"   Features: flex1-5 (5) + accel (3) + gyro (3) = 11 channels")

# Calculate statistics
print("\n📊 Calculating similarity statistics...")

def euclidean_distance(a, b):
    """Simple Euclidean distance between two sequences"""
    if len(a) != len(b):
        indices = np.linspace(0, len(b) - 1, len(a))
        b = np.array([b[int(i)] for i in indices])
    return np.sqrt(np.sum((a - b) ** 2))

# Distance of all trials to template
distances = []
for trial in trials_resampled:
    dist = euclidean_distance(template, trial)
    distances.append(dist)

mean_dist = np.mean(distances)
std_dist = np.std(distances)
max_dist = np.max(distances)

print(f"\n   Mean distance: {mean_dist:.2f}")
print(f"   Std deviation: {std_dist:.2f}")
print(f"   Max distance: {max_dist:.2f}")

# Set threshold (mean + 2*std for ~95% confidence)
threshold = mean_dist + 2 * std_dist
print(f"\n🎯 Detection threshold: {threshold:.2f}")
print(f"   (mean + 2*std for ~95% confidence)")

# Test accuracy on training data
correct = sum(1 for dist in distances if dist < threshold)
accuracy = correct / len(distances) * 100
print(f"\n✅ Training accuracy: {accuracy:.1f}% ({correct}/{len(distances)} trials)")

# Save model
model = {
    'sign_name': 'hello',
    'template': template,
    'threshold': threshold,
    'target_length': TARGET_LENGTH,
    'mean_dist': mean_dist,
    'std_dist': std_dist,
    'max_dist': max_dist,
    'accuracy': accuracy,
    'num_trials': len(trial_ids),
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('hello_sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n💾 Model saved: hello_sign_model.pkl")
print("\n" + "=" * 70)
print(" TRAINING COMPLETE!")
print("=" * 70)
print(f"✅ 'hello' sign model ready for real-time detection")
print(f"📊 Accuracy: {accuracy:.1f}%")
print(f"🎯 Use 'test_hello_realtime.py' to test live gestures!")
print("=" * 70)
