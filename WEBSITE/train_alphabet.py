"""
Alphabet Model Training Script
===============================
Train a machine learning model to recognize sign language alphabets (A-Z)

This script:
1. Loads all alphabet data from the dataset folder
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
DATASET_DIR = r"dataset"  # Folder containing alphabet data

print("="*60)
print("🤖 Alphabet Model Training")
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
print("\n📂 Loading alphabet dataset...")

X = []  # Features
y = []  # Labels

if not os.path.exists(DATASET_DIR):
    print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found!")
    print("Please run 'collect_alphabet_data.py' first to collect data.")
    exit()

# Get all alphabet folders
alphabet_folders = sorted([f for f in os.listdir(DATASET_DIR) 
                          if os.path.isdir(os.path.join(DATASET_DIR, f))])

if len(alphabet_folders) == 0:
    print(f"❌ Error: No alphabet folders found in '{DATASET_DIR}'")
    print("Please collect data using 'collect_alphabet_data.py' first.")
    exit()

print(f"Found {len(alphabet_folders)} alphabet classes: {', '.join(alphabet_folders)}")

total_samples = 0

for label in alphabet_folders:
    folder = os.path.join(DATASET_DIR, label)
    
    # Get all .npy files in this folder
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    print(f"Loading {label}: {len(files)} samples")
    
    for file in files:
        path = os.path.join(folder, file)
        try:
            arr = np.load(path)
            fixed = fix_shape(arr)
            X.append(fixed)
            y.append(label)
            total_samples += 1
        except Exception as e:
            print(f"⚠️ Warning: Could not load {path}: {e}")

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

# Encode labels (A-Z → 0-25)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Classes: {list(le.classes_)}")
print(f"Number of classes: {len(le.classes_)}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2,  # 20% for testing
    random_state=42,
    stratify=y_encoded  # Ensure balanced split
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
print("\n🏋️ Training Neural Network model...")
print("This may take a few minutes...\n")

# Create and train the model
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
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

# Show confusion matrix summary
cm = confusion_matrix(y_test, y_pred)
print("\n" + "="*60)
print("Confusion Matrix Summary:")
print("="*60)
print(f"Correctly classified: {np.trace(cm)} / {np.sum(cm)}")
print(f"Misclassified: {np.sum(cm) - np.trace(cm)} / {np.sum(cm)}")

# Find most confused pairs
if len(le.classes_) > 1:
    print("\nMost confused letter pairs:")
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((cm[i][j], le.classes_[i], le.classes_[j]))
    
    confused_pairs.sort(reverse=True)
    for count, true_label, pred_label in confused_pairs[:5]:
        print(f"  {true_label} → {pred_label}: {count} times")

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
print("\n💾 Saving model...")

joblib.dump(model, "alphabet_model.pkl")
joblib.dump(le, "alphabet_labels.pkl")

print("✅ Saved: alphabet_model.pkl")
print("✅ Saved: alphabet_labels.pkl")

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"Model: Neural Network with 3 hidden layers")
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy: {test_acc * 100:.2f}%")
print(f"Total Classes: {len(le.classes_)}")
print(f"Total Samples: {total_samples}")
print("\n📝 Next steps:")
print("   1. If accuracy is low, collect more data for weak letters")
print("   2. Run 'python sign_language_detector.py' to test the model")
print("="*60)
