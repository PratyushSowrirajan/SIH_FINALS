"""
Setup and installation helper for Facial Expression Recognition System
Run this script to verify and install all dependencies
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor} OK")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      capture_output=True, check=True)
        print("✓ pip is available")
        return True
    except:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                      check=True)
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying imports...")
    
    packages = {
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_ok = False
    
    return all_ok

def check_camera():
    """Check if camera/webcam is available."""
    print("\nChecking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera is working")
                return True
        print("⚠ Camera not detected or not working")
        return False
    except Exception as e:
        print(f"⚠ Could not test camera: {e}")
        return False

def create_default_config():
    """Create default expressions.json if it doesn't exist."""
    import json
    config_file = 'expressions.json'
    
    if os.path.exists(config_file):
        print(f"✓ {config_file} already exists")
        return True
    
    default_expressions = {
        "neutral": {
            "mouth_openness": [0.01, 0.15],
            "avg_eye_openness": [0.08, 0.25],
            "avg_eyebrow_raise": [0.01, 0.15]
        },
        "happy": {
            "mouth_openness": [0.08, 0.5],
            "mouth_width": [0.3, 1.0],
            "avg_eye_openness": [0.05, 0.30],
            "lip_corner_left_elevation": [0.05, float('inf')],
            "lip_corner_right_elevation": [0.05, float('inf')]
        },
        "sad": {
            "mouth_openness": [0.01, 0.15],
            "mouth_aspect_ratio": [0.01, 0.08],
            "avg_eyebrow_raise": [0.05, float('inf')],
            "lip_corner_left_elevation": [-float('inf'), -0.01],
            "lip_corner_right_elevation": [-float('inf'), -0.01]
        },
        "surprised": {
            "mouth_openness": [0.15, 0.6],
            "avg_eye_openness": [0.20, float('inf')],
            "avg_eyebrow_raise": [0.15, float('inf')]
        },
        "angry": {
            "mouth_openness": [0.01, 0.20],
            "avg_eye_openness": [0.05, 0.20],
            "avg_eyebrow_raise": [-0.10, 0.01],
            "nostril_flare": [0.08, float('inf')]
        },
        "disgusted": {
            "mouth_openness": [0.05, 0.25],
            "mouth_aspect_ratio": [0.05, 0.15],
            "nostril_flare": [0.07, float('inf')],
            "avg_eye_openness": [0.05, 0.18]
        }
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(default_expressions, f, indent=2)
        print(f"✓ Created default {config_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create {config_file}: {e}")
        return False

def main():
    """Run all setup checks."""
    print("=" * 50)
    print("Facial Expression Recognition - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("pip Package Manager", check_pip),
        ("Package Installation", install_requirements),
        ("Import Verification", verify_imports),
        ("Default Configuration", create_default_config),
        ("Camera Detection", check_camera),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n[*] {check_name}...", end=" ")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Setup Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ Setup complete! Run with: python main.py")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
