# Installation & Verification Checklist

Complete this checklist to ensure your Facial Expression Recognition System is ready to use.

## ✅ Pre-Installation

- [ ] Python 3.8+ installed
  ```bash
  python --version
  ```
  Expected output: `Python 3.x.x` where x >= 8

- [ ] pip package manager available
  ```bash
  pip --version
  ```
  Expected output: `pip x.x.x from...`

- [ ] Webcam/camera connected and working
  - Test with other applications
  - Check device manager for camera device

- [ ] Project folder created
  ```
  c:\Users\Pratyush Sowrirajan\Desktop\face reco\
  ```

## 📦 Installation Steps

### Step 1: Navigate to Project Directory
```bash
cd "c:\Users\Pratyush Sowrirajan\Desktop\face reco"
```

- [ ] Confirmed in correct directory
- [ ] All Python files visible (main.py, etc.)

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

- [ ] OpenCV installed (4.8.1.78)
  ```bash
  python -c "import cv2; print(cv2.__version__)"
  ```

- [ ] MediaPipe installed (0.10.3)
  ```bash
  python -c "import mediapipe; print(mediapipe.__version__)"
  ```

- [ ] NumPy installed (1.24.3)
  ```bash
  python -c "import numpy; print(numpy.__version__)"
  ```

### Step 3: Verify Installation
```bash
python setup.py
```

- [ ] All checks passed (6/6)
- [ ] Camera detected and working
- [ ] expressions.json created

## 🧪 Testing

### Test 1: Import Check
```bash
python -c "from expression_detector import ExpressionDetector; print('✓ Ready')"
```
- [ ] Prints "✓ Ready"

### Test 2: Camera Access
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('✓ Camera OK' if cap.isOpened() else '✗ Camera Failed'); cap.release()"
```
- [ ] Prints "✓ Camera OK"

### Test 3: Run Main Application
```bash
python main.py
```

Verification steps:
- [ ] Webcam window opens
- [ ] Shows "Facial Expression Recognition" title
- [ ] Camera feed visible (mirrored)
- [ ] Displays detected expression
- [ ] Shows confidence bar
- [ ] Updates in real-time (30 FPS)

Controls to test:
- [ ] Press 'Q' - Application quits
- [ ] Press 'S' - Screenshot saved
- [ ] Press 'C' - Configuration menu appears

### Test 4: Expression Detection
While running `main.py`:

- [ ] **Neutral Face** - Detects "neutral"
- [ ] **Smile** - Detects "happy"
- [ ] **Frown** - Detects "sad"
- [ ] **Wide eyes** - Detects "surprised"
- [ ] **Furrowed brow** - Detects "angry"

### Test 5: Calibration Tool
```bash
python configure_expressions.py
```

- [ ] Menu displays with 5 options
- [ ] Can select "Calibrate new expression"
- [ ] Can collect samples (press SPACE)
- [ ] Expression saved to expressions.json

## 🎨 Functionality Verification

### Expression Detection
- [ ] Happy expression: ✓ Detected (green)
- [ ] Sad expression: ✓ Detected (blue)
- [ ] Angry expression: ✓ Detected (red)
- [ ] Surprised expression: ✓ Detected (orange)
- [ ] Neutral expression: ✓ Detected (gray)
- [ ] Disgusted expression: ✓ Detected (purple)

### Confidence Scoring
- [ ] Confidence bar updates smoothly
- [ ] Shows percentage (0-100%)
- [ ] Higher for exaggerated expressions
- [ ] Decreases when relaxing face

### UI Elements
- [ ] Expression name displayed
- [ ] Confidence visualization works
- [ ] Face mesh landmarks visible (dots)
- [ ] Color changes with expression
- [ ] FPS indicator shows (if enabled)

## 📊 Performance Verification

### FPS Test
```bash
python main.py
# Watch the frame rate indicator
```

- [ ] Maintains 25+ FPS
- [ ] No lag in UI updates
- [ ] Smooth video playback

### Latency Test
- [ ] Expression update: <100ms delay
- [ ] Confidence bar: Smooth updates
- [ ] No stuttering or freezing

### Resource Usage
```bash
# While running python main.py
# Open Task Manager (Ctrl+Shift+Esc)
```

- [ ] CPU usage: 15-25%
- [ ] Memory: ~200-300 MB
- [ ] No memory leaks over time

## 🔧 Configuration Verification

### Check expressions.json
```bash
python -c "import json; f = json.load(open('expressions.json')); print(f'Expressions: {list(f.keys())}')"
```

- [ ] File exists and is valid JSON
- [ ] Contains 6+ expressions
- [ ] Each expression has thresholds

### Save Custom Expression
1. Run `python main.py`
2. Press 'C'
3. Enter "test" as expression name
4. Follow prompts

- [ ] Custom expression saved
- [ ] Can be detected in real-time
- [ ] Appears in expressions.json

## 📁 File Verification

```
Project Directory:
├── ✓ main.py (executable)
├── ✓ expression_detector.py (source)
├── ✓ configure_expressions.py (executable)
├── ✓ setup.py (executable)
├── ✓ expressions.json (config)
├── ✓ requirements.txt (dependencies)
├── ✓ README.md (documentation)
├── ✓ QUICKSTART.md (quick guide)
├── ✓ IMPLEMENTATION.md (technical)
├── ✓ EXAMPLES.md (code samples)
└── ✓ PROJECT_SUMMARY.md (overview)
```

- [ ] All 11 files present
- [ ] No missing files
- [ ] All readable and not corrupted

## 🎓 Documentation Review

- [ ] Read QUICKSTART.md (5 min)
- [ ] Read README.md sections 1-3 (10 min)
- [ ] Understand built-in expressions (5 min)
- [ ] Review EXAMPLES.md for use cases (5 min)

## 🚀 Ready to Use Checklist

- [ ] All dependencies installed
- [ ] Main application runs without errors
- [ ] Camera works and is detected
- [ ] Expressions are detected accurately
- [ ] Confidence scores display
- [ ] Custom expressions can be saved
- [ ] Performance is acceptable (25+ FPS)
- [ ] UI is clean and readable
- [ ] No error messages in console

## 🎯 Success Criteria

✅ **You're ready if:**

1. ✓ `python main.py` runs without errors
2. ✓ Webcam feed displays in real-time
3. ✓ Expression detection works for all faces
4. ✓ Confidence scores are displayed
5. ✓ System runs at 25+ FPS with <100ms latency
6. ✓ Custom expressions can be trained
7. ✓ UI is responsive and looks good
8. ✓ Documentation is clear and helpful

## 🆘 Troubleshooting Reference

| Problem | Solution | Check |
|---------|----------|-------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` | ✓ Dependencies |
| Camera not detected | Check camera in Device Manager | ✓ Hardware |
| Slow FPS | Reduce resolution, close apps | ✓ Performance |
| Expression not detected | Run calibration tool | ✓ Configuration |
| Import errors | Update packages | ✓ Versions |

See README.md troubleshooting section for more.

## 📝 First Run Checklist

- [ ] Dependencies installed
- [ ] Project folder setup
- [ ] Run `python main.py`
- [ ] Verify camera works
- [ ] Test 2-3 expressions
- [ ] Check FPS (should be 25+)
- [ ] Press 'Q' to quit
- [ ] Run `python configure_expressions.py` to test calibration

## ✨ After Installation

**Next Steps:**

1. ✅ Try the main app for 5 minutes
2. ✅ Calibrate 2-3 custom expressions
3. ✅ Adjust expressions.json thresholds
4. ✅ Review code examples in EXAMPLES.md
5. ✅ Integrate into your own project

## 📞 Need Help?

1. Check QUICKSTART.md for quick setup
2. See README.md for detailed info
3. Review EXAMPLES.md for code samples
4. Check IMPLEMENTATION.md for technical details
5. Run `python setup.py` to verify everything

## ✅ Verification Complete!

Once you've checked all items above, your Facial Expression Recognition System is:

- ✓ Properly installed
- ✓ Fully functional
- ✓ Ready for production
- ✓ Well documented
- ✓ Optimized for performance

**Run `python main.py` to get started!** 🚀

---

**Date Completed:** _______________  
**System:** Windows / macOS / Linux (circle one)  
**Python Version:** _______________  
**Notes:** _______________________________________________________________

