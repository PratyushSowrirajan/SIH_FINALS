"""
ISL Sentence Data Collection
=============================
Collect sign language data for common ISL sentences/phrases

Reference: https://youtu.be/VtbYvVDItvg
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------------------------------------------------
# COMMON ISL SENTENCES
# ---------------------------------------------------------
SENTENCES = [
    "HELLO",
    "GOOD_MORNING",
    "GOOD_AFTERNOON", 
    "GOOD_EVENING",
    "GOOD_NIGHT",
    "THANK_YOU",
    "WELCOME",
    "SORRY",
    "HOW_ARE_YOU",
    "I_AM_FINE",
    "WHAT_IS_YOUR_NAME",
    "MY_NAME_IS",
    "NICE_TO_MEET_YOU",
    "PLEASE",
    "YES",
    "NO",
    "HELP",
    "EXCUSE_ME",
    "GOODBYE",
    "SEE_YOU_LATER"
]

# Descriptions for each sentence
SENTENCE_DESCRIPTIONS = {
    "HELLO": "Wave hand with smile",
    "GOOD_MORNING": "Touch forehead, then sweep hand forward",
    "GOOD_AFTERNOON": "Touch chest, then sweep hand forward",
    "GOOD_EVENING": "Touch shoulder, then sweep hand forward",
    "GOOD_NIGHT": "Touch forehead, close eyes gesture",
    "THANK_YOU": "Flat hand from chin forward",
    "WELCOME": "Open arms gesture, palms up",
    "SORRY": "Fist on chest, circular motion",
    "HOW_ARE_YOU": "Point forward, tap chest twice",
    "I_AM_FINE": "Thumbs up, tap chest",
    "WHAT_IS_YOUR_NAME": "Point forward, tap side of head",
    "MY_NAME_IS": "Point to self, tap side of head",
    "NICE_TO_MEET_YOU": "Handshake gesture in air",
    "PLEASE": "Flat hand on chest, circular motion",
    "YES": "Fist nod up and down",
    "NO": "Index and middle finger tap together",
    "HELP": "Thumbs up, place on flat palm",
    "EXCUSE_ME": "Tap shoulder area",
    "GOODBYE": "Wave hand left to right",
    "SEE_YOU_LATER": "Point to eyes, then point forward"
}

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

DATASET_DIR = "sentence_data"
TARGET_SAMPLES = 200

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_landmarks(results):
    """Extract hand landmarks"""
    all_points = []
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                all_points.extend([lm.x, lm.y, lm.z])
    
    return np.array(all_points)

def collect_sentence(sentence_name):
    """Collect data for one sentence"""
    folder = os.path.join(DATASET_DIR, sentence_name)
    os.makedirs(folder, exist_ok=True)
    
    # Count existing samples
    existing = len([f for f in os.listdir(folder) if f.endswith('.npy')])
    
    if existing >= TARGET_SAMPLES:
        print(f"\n✅ {sentence_name} already has {existing} samples (target: {TARGET_SAMPLES})")
        choice = input("Collect more samples? (y/n): ").strip().lower()
        if choice != 'y':
            return
    
    print("\n" + "="*70)
    print(f"📹 Collecting: {sentence_name}")
    print("="*70)
    print(f"Description: {SENTENCE_DESCRIPTIONS.get(sentence_name, 'No description')}")
    print(f"Existing samples: {existing}")
    print(f"Target: {TARGET_SAMPLES} samples")
    print("\nControls:")
    print("  'S' - Save sample")
    print("  'Q' - Quit this sentence")
    print("="*70)
    
    cap = cv2.VideoCapture(0)
    sample_count = existing
    
    while sample_count < TARGET_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        # Status overlay
        cv2.rectangle(frame, (0, 0), (w, 120), (50, 50, 50), -1)
        
        cv2.putText(frame, f"Sentence: {sentence_name.replace('_', ' ')}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, SENTENCE_DESCRIPTIONS.get(sentence_name, ""), (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        progress = f"Progress: {sample_count}/{TARGET_SAMPLES} ({sample_count*100//TARGET_SAMPLES}%)"
        cv2.putText(frame, progress, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand detection indicator
        if results.multi_hand_landmarks:
            cv2.circle(frame, (w-30, 30), 15, (0, 255, 0), -1)
            cv2.putText(frame, "Press 'S'", (w-120, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (w-30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "No Hand", (w-120, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow(f"Collecting: {sentence_name}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') or key == ord('S'):
            if results.multi_hand_landmarks:
                landmarks = extract_landmarks(results)
                filepath = os.path.join(folder, f"{sample_count + 1}.npy")
                np.save(filepath, landmarks)
                sample_count += 1
                print(f"✓ Saved sample {sample_count}/{TARGET_SAMPLES}")
            else:
                print("✗ No hand detected! Show your hand and try again.")
        
        elif key == ord('q') or key == ord('Q'):
            print(f"\n⏸️  Stopped at {sample_count} samples")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if sample_count >= TARGET_SAMPLES:
        print(f"\n✅ Completed {sentence_name}: {sample_count} samples")
    
    return sample_count

# ---------------------------------------------------------
# MAIN COLLECTION MENU
# ---------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("🎥 ISL SENTENCE DATA COLLECTION")
    print("="*70)
    print(f"Reference Video: https://youtu.be/VtbYvVDItvg")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Target: {TARGET_SAMPLES} samples per sentence")
    print("="*70)
    
    while True:
        print("\n📋 Available Sentences:")
        for i, sentence in enumerate(SENTENCES, 1):
            folder = os.path.join(DATASET_DIR, sentence)
            count = 0
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) if f.endswith('.npy')])
            status = "✅" if count >= TARGET_SAMPLES else "⏳"
            print(f"  {i:2}. {status} {sentence:25} ({count}/{TARGET_SAMPLES} samples)")
        
        print("\n" + "="*70)
        print("Options:")
        print("  1-20  : Collect specific sentence")
        print("  'all' : Collect all sentences in sequence")
        print("  'quit': Exit")
        print("="*70)
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'quit':
            print("\n👋 Collection session ended!")
            break
        
        elif choice == 'all':
            print("\n🚀 Starting collection for ALL sentences...")
            for sentence in SENTENCES:
                collect_sentence(sentence)
                print("\n" + "="*70)
            print("\n🎉 All sentences collected!")
            break
        
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(SENTENCES):
                    collect_sentence(SENTENCES[idx])
                else:
                    print("❌ Invalid number!")
            except ValueError:
                print("❌ Invalid input!")

if __name__ == "__main__":
    main()
