"""
ISL (Indian Sign Language) Complete Alphabet Data Collector
============================================================
Collects training data for ALL 26 letters (A-Z) based on ISL standard.

OFFICIAL REFERENCE VIDEO: https://www.youtube.com/watch?v=qcdivQfA41Y

This script will guide you through collecting high-quality training data
for EVERY letter of the ISL alphabet.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------------------------------------------------
# ISL GESTURE DESCRIPTIONS (Based on Reference Video)
# Video: https://www.youtube.com/watch?v=qcdivQfA41Y
# ---------------------------------------------------------
ISL_GESTURES = {
    'A': 'Closed fist, thumb beside index finger',
    'B': 'All fingers straight up together, thumb across palm',
    'C': 'Hand curved in C-shape',
    'D': 'Index finger up, thumb+middle touch forming circle',
    'E': 'All fingertips touching thumb, curved inward',
    'F': 'Thumb+index form circle (OK sign), other 3 fingers up',
    'G': 'Index+thumb point sideways (like gun)',
    'H': 'Index+middle fingers pointing sideways together',
    'I': 'Only pinky finger extended upward',
    'J': 'Pinky up, draw J motion in air',
    'K': 'Index up, middle angled, thumb between them',
    'L': 'Thumb horizontal (pointing right), index vertical (up) - L shape',
    'M': 'Thumb tucked under first 3 fingers (index, middle, ring)',
    'N': 'Thumb tucked under first 2 fingers (index, middle)',
    'O': 'All fingertips touching to form circular O shape',
    'P': 'Like K but pointing downward',
    'Q': 'Like G but pointing downward',
    'R': 'Index and middle fingers crossed, both up',
    'S': 'Closed fist with thumb covering fingers in front',
    'T': 'Thumb poking out between index and middle fingers',
    'U': 'Index+middle together pointing up (parallel, touching)',
    'V': 'Index+middle separated forming V (peace sign, palm forward)',
    'W': 'Three fingers up and separated (index, middle, ring)',
    'X': 'Index finger bent/hooked at knuckle',
    'Y': 'Thumb and pinky extended (shaka/hang loose sign)',
    'Z': 'Index finger draws letter Z shape in the air'
}

# Timestamps from video (approximate)
ISL_TIMESTAMPS = {
    'A': '0:00-0:03', 'B': '0:04-0:07', 'C': '0:08-0:11', 'D': '0:12-0:15',
    'E': '0:16-0:19', 'F': '0:20-0:23', 'G': '0:24-0:27', 'H': '0:28-0:31',
    'I': '0:32-0:35', 'J': '0:36-0:39', 'K': '0:40-0:43', 'L': '0:44-0:47',
    'M': '0:48-0:51', 'N': '0:52-0:55', 'O': '0:56-0:59', 'P': '1:00-1:03',
    'Q': '1:04-1:07', 'R': '1:08-1:11', 'S': '1:12-1:15', 'T': '1:16-1:19',
    'U': '1:20-1:23', 'V': '1:24-1:27', 'W': '1:28-1:31', 'X': '1:32-1:35',
    'Y': '1:36-1:39', 'Z': '1:40-1:43'
}

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
print("="*80)
print("🇮🇳 ISL (INDIAN SIGN LANGUAGE) COMPLETE ALPHABET DATA COLLECTOR")
print("="*80)
print("\n📹 OFFICIAL REFERENCE: https://www.youtube.com/watch?v=qcdivQfA41Y")
print("\n⚠️  IMPORTANT: This is ISL (Indian Sign Language), NOT ASL!")
print("\n📋 What this script does:")
print("   • Collects training data for ALL 26 letters (A-Z)")
print("   • Shows ISL gesture description for each letter")
print("   • Displays video timestamp reference")
print("   • Guides you through the entire alphabet")
print("\n🎯 Best Practices:")
print("   1. WATCH THE VIDEO FIRST - Study each gesture carefully")
print("   2. Match gestures EXACTLY as shown in the video")
print("   3. Palm should face FORWARD for most letters")
print("   4. Hold gesture steady for 1-2 seconds before saving")
print("   5. Collect 200+ samples per letter for best accuracy")
print("   6. Vary hand position slightly (left, right, up, down)")
print("   7. Keep consistent distance from camera")
print("="*80 + "\n")

SAMPLES_TARGET = 200  # Samples per letter
BASE_DIR = "dataset"

# Ask user which letters to collect
print("📝 Collection Options:")
print("   1. All letters A-Z (recommended for new training)")
print("   2. Specific letters only (for retraining confused letters)")
print("   3. Resume from a specific letter")

choice = input("\nYour choice (1/2/3): ").strip()

ALL_LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

if choice == '2':
    letters_input = input("Enter letters to collect (e.g., L V K R): ").strip().upper()
    LETTERS_TO_COLLECT = [l for l in letters_input.split() if l in ALL_LETTERS]
elif choice == '3':
    start_letter = input("Resume from letter (A-Z): ").strip().upper()
    if start_letter in ALL_LETTERS:
        start_idx = ALL_LETTERS.index(start_letter)
        LETTERS_TO_COLLECT = ALL_LETTERS[start_idx:]
    else:
        print("❌ Invalid letter. Starting from A.")
        LETTERS_TO_COLLECT = ALL_LETTERS
else:
    LETTERS_TO_COLLECT = ALL_LETTERS

if not LETTERS_TO_COLLECT:
    print("❌ No valid letters specified. Exiting.")
    exit()

print(f"\n✅ Will collect data for: {', '.join(LETTERS_TO_COLLECT)}")
print(f"📊 Target: {SAMPLES_TARGET} samples per letter")
print(f"⏱️  Estimated time: {len(LETTERS_TO_COLLECT) * 3-5} minutes")
print("\n🎬 WATCH THE REFERENCE VIDEO NOW!")
print("   https://www.youtube.com/watch?v=qcdivQfA41Y")
input("\nPress ENTER when ready to start collecting...")

# ---------------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------------------------------------------------
# CAMERA SETUP
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

print("✅ Camera opened successfully\n")

# ---------------------------------------------------------
# MAIN COLLECTION LOOP
# ---------------------------------------------------------
for letter_idx, LETTER in enumerate(LETTERS_TO_COLLECT, 1):
    SAVE_DIR = f"{BASE_DIR}/{LETTER}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Count existing samples
    existing_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.npy')]
    count = len(existing_files)
    
    print("\n" + "="*80)
    print(f"📸 LETTER {letter_idx}/{len(LETTERS_TO_COLLECT)}: {LETTER}")
    print("="*80)
    print(f"🎥 Video Timestamp: {ISL_TIMESTAMPS.get(LETTER, 'N/A')}")
    print(f"👋 ISL Gesture: {ISL_GESTURES.get(LETTER, 'Unknown')}")
    print(f"📊 Current: {count} samples | Target: {SAMPLES_TARGET} samples")
    
    if count >= SAMPLES_TARGET:
        skip = input(f"\n✅ Already have {count} samples for '{LETTER}'. Skip? (Y/n): ").strip().lower()
        if skip != 'n':
            print(f"⏭️  Skipping {LETTER}")
            continue
    
    print(f"\n🎯 Show ISL gesture for '{LETTER}' now!")
    print("   Press 'S' to SAVE sample")
    print("   Press 'N' to SKIP to next letter")
    print("   Press 'ESC' to EXIT completely")
    print("="*80 + "\n")
    
    last_save_time = 0
    save_cooldown = 0.3  # Prevent rapid-fire saves
    
    while count < SAMPLES_TARGET:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        landmarks_all = []
        can_save = False
        
        # Extract landmarks
        if results.multi_hand_landmarks:
            can_save = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=3)
                )
                for lm in hand_landmarks.landmark:
                    landmarks_all.extend([lm.x, lm.y, lm.z])
            
            # Pad if only 1 hand
            if len(results.multi_hand_landmarks) == 1:
                landmarks_all.extend([0] * 63)
        
        # ---------------------------------------------------------
        # UI DESIGN
        # ---------------------------------------------------------
        # Reference panel (left side)
        panel_width = 450
        cv2.rectangle(frame, (0, 0), (panel_width, h), (25, 25, 25), -1)
        
        # Header
        cv2.putText(frame, "ISL REFERENCE", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.line(frame, (15, 45), (panel_width-15, 45), (0, 200, 255), 2)
        
        # Current letter - LARGE
        cv2.putText(frame, f"Letter: {LETTER}", (15, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        
        # Alphabet progress
        progress_letters = f"({letter_idx}/{len(LETTERS_TO_COLLECT)} letters)"
        cv2.putText(frame, progress_letters, (15, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        
        # Video timestamp
        timestamp_text = f"Video: {ISL_TIMESTAMPS.get(LETTER, 'N/A')}"
        cv2.putText(frame, timestamp_text, (15, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
        
        # Gesture description (word-wrapped)
        gesture_desc = ISL_GESTURES.get(LETTER, 'Unknown')
        y_offset = 220
        max_width = 40  # characters per line
        
        words = gesture_desc.split(' ')
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if len(test_line) > max_width:
                cv2.putText(frame, current_line.strip(), (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
                y_offset += 30
                current_line = word + " "
            else:
                current_line = test_line
        
        if current_line:
            cv2.putText(frame, current_line.strip(), (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        
        # Sample progress
        progress_pct = (count / SAMPLES_TARGET) * 100
        cv2.putText(frame, f"Samples: {count}/{SAMPLES_TARGET}", (15, h-140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"{progress_pct:.0f}% Complete", (15, h-100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Progress bar
        bar_y = h - 65
        bar_height = 40
        cv2.rectangle(frame, (15, bar_y), (panel_width-15, bar_y+bar_height), (60, 60, 60), -1)
        bar_fill = int((panel_width - 30) * (count / SAMPLES_TARGET))
        if bar_fill > 0:
            cv2.rectangle(frame, (15, bar_y), (15 + bar_fill, bar_y+bar_height), (0, 255, 0), -1)
        
        # Percentage text on bar
        percent_text = f"{int(progress_pct)}%"
        text_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (panel_width - text_size[0]) // 2
        cv2.putText(frame, percent_text, (text_x, bar_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # ---------------------------------------------------------
        # Main camera area status
        # ---------------------------------------------------------
        status_x = panel_width + 30
        
        # Hand detection status
        if can_save and (time.time() - last_save_time) > save_cooldown:
            status_text = "✓ HAND DETECTED - Press 'S' to SAVE"
            status_color = (0, 255, 0)
            cv2.circle(frame, (w-50, 50), 25, (0, 255, 0), -1)
            cv2.putText(frame, "OK", (w-65, 58),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif not can_save:
            status_text = "✗ NO HAND - Show gesture clearly"
            status_color = (0, 0, 255)
            cv2.circle(frame, (w-50, 50), 25, (0, 0, 255), -1)
        else:
            status_text = "⏳ COOLDOWN - Wait a moment..."
            status_color = (0, 165, 255)
            cv2.circle(frame, (w-50, 50), 25, (0, 165, 255), -1)
        
        cv2.putText(frame, status_text, (status_x, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Instructions at bottom
        cv2.rectangle(frame, (0, h-40), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, "S: Save Sample  |  N: Next Letter  |  ESC: Exit", 
                   (status_x, h-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow(f"ISL Alphabet Collector - {LETTER}", frame)
        
        # ---------------------------------------------------------
        # KEYBOARD CONTROLS
        # ---------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n⏹️  Collection stopped by user")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        elif key == ord('n') or key == ord('N'):
            print(f"\n⏭️  Skipping to next letter...")
            break
        
        elif key == ord('s') or key == ord('S'):
            if can_save and (time.time() - last_save_time) > save_cooldown:
                filename = os.path.join(SAVE_DIR, f"{LETTER}_{count}.npy")
                np.save(filename, np.array(landmarks_all))
                count += 1
                last_save_time = time.time()
                print(f"✅ Saved: {LETTER}_{count-1}.npy | Progress: {count}/{SAMPLES_TARGET} ({count/SAMPLES_TARGET*100:.1f}%)")
                
                # Visual flash feedback
                flash = frame.copy()
                cv2.rectangle(flash, (panel_width, 0), (w, h), (0, 255, 0), 30)
                cv2.imshow(f"ISL Alphabet Collector - {LETTER}", flash)
                cv2.waitKey(80)
            else:
                if not can_save:
                    print("⚠️  Cannot save: No hand detected")
                else:
                    print("⚠️  Cannot save: Too fast, wait a moment")
    
    cv2.destroyAllWindows()
    
    if count >= SAMPLES_TARGET:
        print(f"\n🎉 COMPLETED '{LETTER}' - Collected {count} samples!")
    
    # Small break between letters
    if letter_idx < len(LETTERS_TO_COLLECT):
        print(f"\n⏸️  Prepare for next letter...")
        time.sleep(1)

# ---------------------------------------------------------
# CLEANUP & SUMMARY
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*80)
print("🎉 ISL ALPHABET DATA COLLECTION COMPLETE!")
print("="*80)
print("\n📊 Collection Summary:")
print(f"{'Letter':<10} {'Samples':<10} {'Status'}")
print("-" * 40)

total_samples = 0
complete_count = 0

for letter in LETTERS_TO_COLLECT:
    save_dir = f"{BASE_DIR}/{letter}"
    if os.path.exists(save_dir):
        file_count = len([f for f in os.listdir(save_dir) if f.endswith('.npy')])
        total_samples += file_count
        status = "✅ Complete" if file_count >= SAMPLES_TARGET else f"⚠️  Need {SAMPLES_TARGET - file_count} more"
        if file_count >= SAMPLES_TARGET:
            complete_count += 1
        print(f"{letter:<10} {file_count:<10} {status}")

print("-" * 40)
print(f"Total: {total_samples} samples across {len(LETTERS_TO_COLLECT)} letters")
print(f"Complete: {complete_count}/{len(LETTERS_TO_COLLECT)} letters")

print("\n🔄 NEXT STEPS:")
print("="*80)
print("1. Train the model:")
print("   python train_alphabet.py")
print("\n2. Wait for training (2-5 minutes)")
print("\n3. Start the web app:")
print("   python web_app.py")
print("\n4. Test at: http://localhost:5000")
print("="*80 + "\n")

print("💡 TIP: If accuracy is poor for specific letters, run this script")
print("   again with option 2 and recollect those letters with 300+ samples.")
print("\n📹 Reference Video: https://www.youtube.com/watch?v=qcdivQfA41Y\n")
