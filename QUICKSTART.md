# Quick Start Guide

## 1. Installation (30 seconds)

```bash
# Navigate to the project directory
cd "C:\Users\Pratyush Sowrirajan\Desktop\face reco"

# Install dependencies
pip install -r requirements.txt
```

## 2. First Run (10 seconds)

```bash
python main.py
```

Press **Q** to quit, **C** to configure, **S** to save screenshots.

## 3. Configure Your Custom Expressions (Optional)

### Option A: Interactive Configuration (During Runtime)
1. Run `python main.py`
2. Press **C** to configure
3. Enter expression name and parameters
4. Done!

### Option B: Calibration Tool (Recommended)
```bash
python configure_expressions.py
```

Menu:
1. Choose "Calibrate new expression"
2. Make the expression 15+ times while pressing SPACE
3. System auto-generates thresholds
4. Done!

## What You'll See

The application shows:
- ✨ Live webcam feed
- 🎯 Detected facial expression (HAPPY, SAD, ANGRY, etc.)
- 📊 Confidence bar (0-100%)
- 🟢 Face mesh landmarks (minimal, for performance)

## Example Workflow

```
1. python main.py                    # Start app
   ↓
2. Make different facial expressions  # App detects them
   ↓
3. Press 'C' to add custom expression
   ↓
4. Define thresholds or use auto-calibration
   ↓
5. Test your expression               # Real-time detection
```

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | ⚡ Main application - run this |
| `expression_detector.py` | 🧠 Expression detection logic |
| `configure_expressions.py` | ⚙️ Calibration tool |
| `expressions.json` | 💾 Saved expressions |
| `requirements.txt` | 📦 Dependencies |
| `setup.py` | ✅ Verification script |

## Default Expressions (Built-in)

1. **Neutral** - Resting face
2. **Happy** - Smile, cheeks raised
3. **Sad** - Frown, eyebrows down
4. **Surprised** - Eyes wide, mouth open
5. **Angry** - Eyebrows down, nostrils flared
6. **Disgusted** - Nose wrinkled, lip raised

## Tips for Best Performance

✅ **DO:**
- Use good lighting (natural or desk lamp)
- Position camera 30-60cm from face
- Keep face frontal (minimal angles)
- Make clear, exaggerated expressions
- Calibrate for your face type

❌ **DON'T:**
- Use in very dark rooms
- Wear sunglasses or face masks
- Hide facial features with hands
- Move too quickly
- Use very low camera resolution

## Performance Expected

| Metric | Value |
|--------|-------|
| Frame Rate | 25-30 FPS |
| Latency | 30-50ms |
| CPU Usage | 15-25% |
| Memory | ~200MB |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **C** | Configure expression |
| **S** | Save screenshot |

## Troubleshooting

### Camera not working?
```bash
python -c "import cv2; print(cv2.getBuildInfo())"
```

### Low FPS?
- Close background apps
- Reduce resolution in code (change 640x480)
- Update graphics drivers

### Expression not detected?
- Run `python configure_expressions.py` to calibrate
- Ensure good lighting
- Make more exaggerated expressions

### Import errors?
```bash
pip install --upgrade opencv-python mediapipe numpy
```

## Next Steps

1. ✅ Run `main.py` - Test the system
2. ✅ Calibrate 2-3 custom expressions
3. ✅ Adjust thresholds in `expressions.json` for better accuracy
4. ✅ Integrate into your own projects!

## File Locations

- **Config file**: `expressions.json` (auto-created)
- **Screenshots**: Saved in current directory as `screenshot_*.jpg`
- **Log**: Check console output

## Support

Check `README.md` for detailed documentation.

---

**Ready? Run:** `python main.py` 🚀
