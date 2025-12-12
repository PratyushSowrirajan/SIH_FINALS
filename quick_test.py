import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import json

# Load template and config
template = np.load('template_J.npy')
config = json.load(open('template_J_config.json'))

# Load TEST_J data
df = pd.read_csv('TEST_J_dual_dataset_20251212_072221.csv')

features = config['features']
threshold = config['threshold']

print("="*70)
print("TEST_J PREDICTIONS:")
print("="*70)

for trial_id in [1, 2, 3, 4, 5]:
    trial_data = df[df['trial_id'] == trial_id][features].values
    
    # Normalize
    normalized = stats.zscore(trial_data, axis=0, nan_policy='omit')
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    # Calculate DTW distance
    distance, _ = fastdtw(normalized, template, dist=euclidean)
    confidence = 1 / (1 + distance)
    
    # Prediction
    prediction = "J" if distance < threshold else "NOT J"
    
    print(f"\nTrial {trial_id}:")
    print(f"  Prediction: {prediction}")
    print(f"  Distance: {distance:.2f}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

print("\n" + "="*70)
print("Now YOU tell me which trials were actually TRUE J signs!")
