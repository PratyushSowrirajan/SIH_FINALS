"""
Sentence Model Training Script
===============================
Train a machine learning model to recognize sign language sentences/phrases

This script:
1. Loads all sentence data from the sentence_data folder
2. Preprocesses and normalizes the data
3. Trains a Neural Network classifier
4. Saves the trained model and label encoder
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET_DIR = r"sentence_data"  # Folder containing sentence data

print("="*60)
print("🤖 Sentence Model Training")
print("="*60)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def fix_shape(arr):
    """Ensure every sample has shape (126,) - 2 hands × 21 landmarks × 3 coords"""
    arr = arr.flatten()
    
    if len(arr) == 63:  # Only one hand detected
        arr = np.concatenate([arr, np.zeros(63)])  # Pad to 126
    elif len(arr) > 126:
        arr = arr[:126]  # Trim if extra
    elif len(arr) < 126:
        arr = np.pad(arr, (0, 126 - len(arr)))  # Pad to fixed length
    
    return arr

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("\n📂 Loading sentence dataset...")

X = []  # Features
y = []  # Labels

if not os.path.exists(DATASET_DIR):
    print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found!")
    print("Please run 'collect_sentence_data.py' first to collect data.")
    exit()

# Get all sentence folders
sentence_folders = sorted([f for f in os.listdir(DATASET_DIR) 
                          if os.path.isdir(os.path.join(DATASET_DIR, f))])

if len(sentence_folders) == 0:
    print(f"❌ Error: No sentence folders found in '{DATASET_DIR}'")
    print("Please collect data using 'collect_sentence_data.py' first.")
    exit()

print(f"Found {len(sentence_folders)} sentence classes:")
for sentence in sentence_folders:
    print(f"  - {sentence.replace('_', ' ')}")

total_samples = 0

for label in sentence_folders:
    folder = os.path.join(DATASET_DIR, label)
    
    # Get all .npy files in this folder
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    print(f"Loading '{label}': {len(files)} samples")
    
    if len(files) < 50:
        print(f"  ⚠️ Warning: Only {len(files)} samples. Recommend at least 100 for good accuracy.")
    
    for file in files:
        path = os.path.join(folder, file)
        try:
            arr = np.load(path)
            fixed = fix_shape(arr)
            X.append(fixed)
            y.append(label)
            total_samples += 1
        except Exception as e:
            print(f"  ⚠️ Warning: Could not load {path}: {e}")

if total_samples == 0:
    print("❌ Error: No data samples loaded!")
    exit()

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\n✅ Loaded {total_samples} total samples")
print(f"Feature shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ---------------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------------
print("\n🔧 Preprocessing data...")

# Encode labels (sentences → 0, 1, 2, ...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nClasses mapped:")
for idx, sentence in enumerate(le.classes_):
    print(f"  {idx}: {sentence.replace('_', ' ')}")

print(f"\nNumber of classes: {len(le.classes_)}")

# Check if we have enough data
if len(X) < 100:
    print("\n⚠️ WARNING: You have less than 100 total samples.")
    print("   This may result in poor model performance.")
    print("   Recommendation: Collect at least 100-150 samples per sentence.")

# Split into training and testing sets
test_size = 0.2 if len(X) > 50 else 0.15  # Use smaller test set if data is limited

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=test_size,
    random_state=42,
    stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
print("\n🏋️ Training Neural Network model...")
print("This may take a few minutes...\n")

# Adjust model complexity based on dataset size
if len(X) < 200:
    hidden_layers = (128, 64)
    max_iter = 300
else:
    hidden_layers = (256, 128, 64)
    max_iter = 500

print(f"Model architecture: {hidden_layers}")

# Create and train the model
model = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    activation='relu',
    solver='adam',
    max_iter=max_iter,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------
print("\n📊 Evaluating model...")

# Training accuracy
train_acc = model.score(X_train, y_train)
print(f"Training Accuracy: {train_acc * 100:.2f}%")

# Testing accuracy
test_acc = model.score(X_test, y_test)
print(f"Testing Accuracy: {test_acc * 100:.2f}%")

# Detailed classification report
y_pred = model.predict(X_test)
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n" + "="*60)
print("Confusion Matrix Summary:")
print("="*60)
print(f"Correctly classified: {np.trace(cm)} / {np.sum(cm)}")
print(f"Misclassified: {np.sum(cm) - np.trace(cm)} / {np.sum(cm)}")

# Find most confused pairs
if len(le.classes_) > 1:
    print("\nMost confused sentence pairs:")
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((
                    cm[i][j], 
                    le.classes_[i].replace('_', ' '), 
                    le.classes_[j].replace('_', ' ')
                ))
    
    confused_pairs.sort(reverse=True)
    if confused_pairs:
        for count, true_label, pred_label in confused_pairs[:5]:
            print(f"  '{true_label}' → '{pred_label}': {count} times")
    else:
        print("  No confusion - perfect predictions!")

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
print("\n💾 Saving model...")

joblib.dump(model, "sentence_model.pkl")
joblib.dump(le, "sentence_labels.pkl")

print("✅ Saved: sentence_model.pkl")
print("✅ Saved: sentence_labels.pkl")

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"Model: Neural Network with {len(hidden_layers)} hidden layers")
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy: {test_acc * 100:.2f}%")
print(f"Total Classes: {len(le.classes_)}")
print(f"Total Samples: {total_samples}")

# Provide recommendations
if test_acc < 0.8:
    print("\n⚠️ Accuracy is below 80%. Recommendations:")
    print("   - Collect more samples (aim for 150+ per sentence)")
    print("   - Ensure consistent gestures during data collection")
    print("   - Check if gestures are too similar between sentences")
elif test_acc < 0.9:
    print("\n✅ Good accuracy! To improve further:")
    print("   - Collect more samples for better generalization")
    print("   - Vary hand positions during data collection")
else:
    print("\n🎉 Excellent accuracy!")

print("\n📝 Next steps:")
print("   1. Collect more sentences if needed")
print("   2. Run 'python sign_language_detector.py' to test both models")
print("="*60)
