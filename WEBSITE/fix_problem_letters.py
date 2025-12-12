"""
Quick Fix: Recollect Specific Problem Letters
==============================================
Use this to quickly recollect data for letters that aren't detecting correctly.

For letters like O, C, I, E that have similar hand shapes.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------------------------------------------------
# PROBLEM LETTERS TO FIX
# ---------------------------------------------------------
PROBLEM_LETTERS = ['O', 'C', 'I', 'E']  # Edit this list as needed
SAMPLES_TARGET = 300  # More samples for problem letters
BASE_DIR = "dataset"

# ISL Gesture descriptions
ISL_GESTURES = {
    'O': 'ALL fingertips touch thumb forming PERFECT CIRCLE - palm forward',
    'C': 'Hand curved like letter C - thumb and fingers apart forming curve',
    'I': 'ONLY PINKY finger up - all others folded down completely',
    'E': 'Fingertips bent touching thumb - like holding small ball',
}

print("="*70)
print("🔧 QUICK FIX: Recollect Problem Letters")
print("="*70)
print(f"\n📝 Letters to fix: {', '.join(PROBLEM_LETTERS)}")
print(f"🎯 Target: {SAMPLES_TARGET} samples per letter (more = better)")
print("\n⚠️  CRITICAL TIPS FOR THESE LETTERS:")
print("   O: Make a TIGHT circle, all fingertips touching")
print("   C: Wide curve, thumb and fingers DON'T touch")
print("   I: ONLY pinky up, make it VERY clear")
print("   E: Fingertips curl inward touching thumb tip")
print("\n💡 For EACH letter:")
print("   1. Study the ISL video at the timestamp")
print("   2. Make gesture VERY DISTINCT")
print("   3. Hold steady 2 seconds before saving")
print("   4. Vary position (up, down, left, right)")
print("   5. Collect MORE samples than before")
print("="*70 + "\n")

input("Press ENTER to start...")

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
# CAMERA
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

# ---------------------------------------------------------
# COLLECTION LOOP
# ---------------------------------------------------------
for LETTER in PROBLEM_LETTERS:
    SAVE_DIR = f"{BASE_DIR}/{LETTER}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Count existing
    existing = [f for f in os.listdir(SAVE_DIR) if f.endswith('.npy')]
    count = len(existing)
    
    print("\n" + "="*70)
    print(f"📸 FIXING LETTER: {LETTER}")
    print("="*70)
    print(f"Current samples: {count}")
    print(f"Target: {SAMPLES_TARGET}")
    print(f"ISL Gesture: {ISL_GESTURES.get(LETTER, 'See video')}")
    print("\n🎯 Make this gesture VERY CLEAR and DISTINCT!")
    print("="*70 + "\n")
    
    last_save = 0
    cooldown = 0.3
    
    while count < SAMPLES_TARGET:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        landmarks_all = []
        can_save = False
        
        if results.multi_hand_landmarks:
            can_save = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=3)
                )
                for lm in hand_landmarks.landmark:
                    landmarks_all.extend([lm.x, lm.y, lm.z])
            
            if len(results.multi_hand_landmarks) == 1:
                landmarks_all.extend([0] * 63)
        
        # UI
        panel_w = 420
        cv2.rectangle(frame, (0, 0), (panel_w, h), (30, 30, 30), -1)
        
        # Title
        cv2.putText(frame, f"FIXING: {LETTER}", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 255), 3)
        
        # Gesture
        gesture = ISL_GESTURES.get(LETTER, '')
        y = 120
        for line in [gesture[i:i+35] for i in range(0, len(gesture), 35)]:
            cv2.putText(frame, line, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            y += 30
        
        # Progress
        pct = (count / SAMPLES_TARGET) * 100
        cv2.putText(frame, f"{count}/{SAMPLES_TARGET} ({pct:.0f}%)", (15, h-100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Progress bar
        bar_y = h - 60
        cv2.rectangle(frame, (15, bar_y), (panel_w-15, bar_y+35), (60, 60, 60), -1)
        bar_fill = int((panel_w - 30) * (count / SAMPLES_TARGET))
        if bar_fill > 0:
            cv2.rectangle(frame, (15, bar_y), (15 + bar_fill, bar_y+35), (0, 255, 0), -1)
        
        # Status
        if can_save and (time.time() - last_save) > cooldown:
            status = "✓ READY - Press 'S'"
            color = (0, 255, 0)
            cv2.circle(frame, (w-50, 50), 30, (0, 255, 0), -1)
        else:
            status = "Show gesture clearly" if not can_save else "Wait..."
            color = (0, 0, 255) if not can_save else (255, 165, 0)
            cv2.circle(frame, (w-50, 50), 30, color, -1)
        
        cv2.putText(frame, status, (panel_w + 20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Instructions
        cv2.rectangle(frame, (0, h-35), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, "S: Save | N: Next Letter | ESC: Exit", 
                   (panel_w + 20, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow(f"Quick Fix - {LETTER}", frame)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        elif key == ord('n') or key == ord('N'):
            print(f"\n⏭️  Skipping {LETTER}...")
            break
        
        elif key == ord('s') or key == ord('S'):
            if can_save and (time.time() - last_save) > cooldown:
                filename = os.path.join(SAVE_DIR, f"{LETTER}_{count}.npy")
                np.save(filename, np.array(landmarks_all))
                count += 1
                last_save = time.time()
                print(f"✅ {LETTER}_{count-1} saved | Progress: {count}/{SAMPLES_TARGET} ({pct:.0f}%)")
                
                # Flash
                flash = frame.copy()
                cv2.rectangle(flash, (panel_w, 0), (w, h), (0, 255, 0), 25)
                cv2.imshow(f"Quick Fix - {LETTER}", flash)
                cv2.waitKey(80)
    
    cv2.destroyAllWindows()
    print(f"\n✅ Completed {LETTER} with {count} samples!")
    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("🎉 RECOLLECTION COMPLETE!")
print("="*70)
print("\n📊 Summary:")
for letter in PROBLEM_LETTERS:
    path = f"{BASE_DIR}/{letter}"
    if os.path.exists(path):
        cnt = len([f for f in os.listdir(path) if f.endswith('.npy')])
        status = "✅" if cnt >= SAMPLES_TARGET else "⚠️"
        print(f"   {status} {letter}: {cnt} samples")

print("\n🔄 NEXT STEPS:")
print("1. python train_alphabet.py")
print("2. python web_app.py")
print("3. Test O, C, I, E letters!")
print("="*70 + "\n")
