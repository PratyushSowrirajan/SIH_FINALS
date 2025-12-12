"""
DTW Template Matching for Sign Language Recognition
Uses Dynamic Time Warping with template matching for gesture detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

print("="*70)
print(" DTW TEMPLATE MATCHING - SIGN LANGUAGE RECOGNITION")
print("="*70)

# Load dataset
print("\n📁 Loading J sign dataset...")
df = pd.read_csv('J_dual_dataset_20251212_063557.csv')

print(f"   Total rows: {len(df)}")
print(f"   Trials: {sorted(df['trial_id'].unique())}")
print(f"   Total trials: {df['trial_id'].nunique()}")

# Select only 12 MPU features (ignore flex sensors)
mpu_features = [
    'ax_right', 'ay_right', 'az_right', 'gx_right', 'gy_right', 'gz_right',  # RIGHT glove
    'ax_left', 'ay_left', 'az_left', 'gx_left', 'gy_left', 'gz_left'   # LEFT glove
]

print(f"\n🎯 Using 12 MPU features: {mpu_features}")

# Extract trials as separate sequences
print("\n📊 Extracting individual trials...")
trials = []
trial_ids = []

for trial_id in sorted(df['trial_id'].unique()):
    trial_data = df[df['trial_id'] == trial_id][mpu_features].values
    
    # Check if trial has data
    if len(trial_data) > 10:  # minimum 10 samples
        trials.append(trial_data)
        trial_ids.append(trial_id)
        print(f"   Trial {trial_id}: {len(trial_data)} samples")

print(f"\n✅ Loaded {len(trials)} valid trials")

# Normalize each trial (z-score normalization)
print("\n🔧 Normalizing features (z-score)...")
normalized_trials = []

for i, trial in enumerate(trials):
    # Normalize each feature column independently
    normalized = stats.zscore(trial, axis=0, nan_policy='omit')
    
    # Handle any NaN values (from constant columns)
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    normalized_trials.append(normalized)
    print(f"   Trial {trial_ids[i]}: normalized shape {normalized.shape}")

print(f"\n✅ All trials normalized")

# Create template using DTW Barycenter Averaging
print("\n🧠 Creating DTW template...")
print("   Using median-length trial as template (simple & effective)")

# Use the median-length trial as template
lengths = [len(t) for t in normalized_trials]
median_idx = np.argsort(lengths)[len(lengths)//2]
template = normalized_trials[median_idx]

print(f"   ✅ Using trial {trial_ids[median_idx]} as template")
print(f"   Template length: {len(template)} samples")

# Calculate DTW distances from each trial to template
print("\n📏 Calculating DTW distances from all trials to template...")
distances = []

for i, trial in enumerate(normalized_trials):
    # Calculate multivariate DTW distance using fastdtw
    distance, _ = fastdtw(trial, template, dist=euclidean)
    distances.append(distance)
    print(f"   Trial {trial_ids[i]}: distance = {distance:.4f}")

distances = np.array(distances)

# Calculate threshold
mean_dist = np.mean(distances)
std_dist = np.std(distances)
threshold = mean_dist + 2 * std_dist

print(f"\n📊 Distance Statistics:")
print(f"   Mean distance: {mean_dist:.4f}")
print(f"   Std deviation: {std_dist:.4f}")
print(f"   Min distance: {np.min(distances):.4f}")
print(f"   Max distance: {np.max(distances):.4f}")
print(f"\n✅ Detection threshold: {threshold:.4f}")
print(f"   (Any new gesture with distance < {threshold:.4f} is classified as 'J')")

# Test accuracy on training data
print("\n🧪 Testing on training data...")
correct = 0
for i, dist in enumerate(distances):
    predicted = "J" if dist < threshold else "NOT J"
    confidence = 1 / (1 + dist)
    
    if predicted == "J":
        correct += 1
    
    status = "✓" if predicted == "J" else "✗"
    print(f"   {status} Trial {trial_ids[i]}: {predicted} (confidence: {confidence:.2%}, distance: {dist:.4f})")

accuracy = correct / len(distances) * 100
print(f"\n📈 Training accuracy: {accuracy:.1f}% ({correct}/{len(distances)} trials)")

# Save template and threshold
print("\n💾 Saving template and configuration...")
np.save('template_J.npy', template)

config = {
    'sign': 'J',
    'threshold': threshold,
    'mean_distance': mean_dist,
    'std_distance': std_dist,
    'num_trials': len(trials),
    'features': mpu_features
}

import json
with open('template_J_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"   ✅ Template saved to: template_J.npy")
print(f"   ✅ Config saved to: template_J_config.json")

print("\n" + "="*70)
print("✅ DTW TEMPLATE TRAINING COMPLETE!")
print("="*70)
print(f"\nSign: J")
print(f"Template shape: {template.shape}")
print(f"Detection threshold: {threshold:.4f}")
print(f"Training accuracy: {accuracy:.1f}%")
print("\nNext step: Create templates for other signs!")
