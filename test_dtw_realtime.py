"""
Real-time DTW Testing - Collect NEW gesture and test against trained template
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import json

print("="*70)
print(" REAL-TIME DTW TESTING - UNSEEN DATA")
print("="*70)

# Load trained template
print("\n📁 Loading trained template for sign 'J'...")
template = np.load('template_J.npy')

with open('template_J_config.json', 'r') as f:
    config = json.load(f)

print(f"   ✅ Template loaded")
print(f"   Detection threshold: {config['threshold']:.4f}")
print(f"   Trained on {config['num_trials']} trials")

# Ask user to collect NEW test data
print("\n" + "="*70)
print(" COLLECT NEW TEST DATA")
print("="*70)
print("\n🎯 Collect TEST data with sign name 'TEST_J':")
print("   - Perform as many trials as you want")
print("   - RANDOMLY do TRUE J sign OR different gesture")
print("   - System will predict BLINDLY")
print("   - You verify if predictions are correct")

test_sign = "TEST_J"
print(f"\n   Run: python collect_dual_gloves.py")
print(f"   Sign name: {test_sign}")
print(f"   Trials: As many as you want!")

input("\n⏸️  Press Enter AFTER you've collected the test data...")

# Find test dataset
import glob
test_files = glob.glob(f"*{test_sign}*.csv")

if not test_files:
    print(f"\n❌ No test dataset found with name '{test_sign}'")
    print(f"   Please collect data first!")
    exit(1)

# Use most recent file
test_file = sorted(test_files)[-1]
print(f"\n📊 Found test file: {test_file}")

# Load test data
df_test = pd.read_csv(test_file)
print(f"   Rows: {len(df_test)}")
print(f"   Trials: {sorted(df_test['trial_id'].unique())}")

# Test each trial
print("\n" + "="*70)
print(" TESTING AGAINST TRAINED TEMPLATE")
print("="*70)

mpu_features = config['features']
threshold = config['threshold']

results = []

for trial_id in sorted(df_test['trial_id'].unique()):
    trial_data = df_test[df_test['trial_id'] == trial_id]
    
    # Extract and normalize features
    sequence = trial_data[mpu_features].values
    normalized = stats.zscore(sequence, axis=0, nan_policy='omit')
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    # Calculate DTW distance
    distance, _ = fastdtw(normalized, template, dist=euclidean)
    confidence = 1 / (1 + distance)
    
    # Prediction (BLIND - system doesn't know ground truth)
    is_match = distance < threshold
    prediction = "J" if is_match else "NOT J"
    
    results.append({
        'trial': trial_id,
        'prediction': prediction,
        'distance': distance,
        'confidence': confidence,
        'is_match': is_match
    })
    
    status = "✓" if is_match else "✗"
    
    print(f"\n{status} Trial {trial_id}:")
    print(f"   🤖 PREDICTION: {prediction}")
    print(f"   📏 Distance: {distance:.4f}")
    print(f"   🎯 Threshold: {threshold:.4f}")
    print(f"   💯 Confidence: {confidence:.2%}")
    print(f"   Result: {'RECOGNIZED as J' if is_match else 'REJECTED (not J)'}")

# Summary
print("\n" + "="*70)
print(" BLIND TEST RESULTS SUMMARY")
print("="*70)

recognized = sum(1 for r in results if r['is_match'])
rejected = sum(1 for r in results if not r['is_match'])
total = len(results)

print(f"\n📊 Predictions Made (BLIND):")
print(f"   ✅ Recognized as 'J': {recognized}/{total} trials")
print(f"   ❌ Rejected (not J): {rejected}/{total} trials")

# Show all predictions
print(f"\n🎯 All Predictions:")
for r in results:
    pred_symbol = "✓" if r['is_match'] else "✗"
    print(f"   {pred_symbol} Trial {r['trial']}: {r['prediction']} (distance: {r['distance']:.2f}, confidence: {r['confidence']:.1%})")

# Distance statistics
distances = [r['distance'] for r in results]
print(f"\n📏 Distance Statistics:")
print(f"   Min: {min(distances):.2f}")
print(f"   Max: {max(distances):.2f}")
print(f"   Avg: {np.mean(distances):.2f}")
print(f"   Threshold: {threshold:.2f}")

print("\n" + "="*70)
print("⚠️  NOW YOU VERIFY:")
print("="*70)
print("\nFor each trial, tell me:")
print("  - Was the prediction CORRECT or WRONG?")
print("  - Did you actually perform J or something else?")
print("\nThis is the REAL test with no rigging!")
