"""
TruthLens v4.0 — Setup Checker
Run this FIRST to verify everything is installed correctly.

Usage:
  python check_setup.py
"""

import sys
import os

print("\n" + "="*50)
print("  TruthLens v4.0 — Setup Checker")
print("="*50 + "\n")

all_ok = True

# 1. Python version
print("Checking Python version...")
version = sys.version_info
if version.major == 3 and version.minor == 10:
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"  ⚠️  Python {version.major}.{version.minor}.{version.micro} "
          f"(recommended: 3.10.11)")

# 2. Check all packages
print("\nChecking packages...")
packages = [
    ('flask',             'Flask'),
    ('flask_cors',        'Flask-CORS'),
    ('flask_limiter',     'Flask-Limiter'),
    ('flask_socketio',    'Flask-SocketIO'),
    ('torch',             'PyTorch'),
    ('torchvision',       'TorchVision'),
    ('timm',              'TIMM'),
    ('cv2',               'OpenCV'),
    ('PIL',               'Pillow'),
    ('numpy',             'NumPy'),
    ('sklearn',           'Scikit-learn'),
    ('librosa',           'Librosa'),
    ('soundfile',         'SoundFile'),
    ('requests',          'Requests'),
    ('tqdm',              'TQDM'),
]

for module, name in packages:
    try:
        pkg = __import__(module)
        ver = getattr(pkg, '__version__', 'unknown')
        print(f"  ✅ {name:20} {ver}")
    except ImportError:
        print(f"  ❌ {name:20} NOT INSTALLED")
        all_ok = False

# 3. Check PyTorch device
print("\nChecking PyTorch device...")
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"  ✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB")
    else:
        print(f"  ℹ️  CPU only (training will be slower)")
except Exception as e:
    print(f"  ❌ PyTorch check failed: {e}")

# 4. Check folders exist
print("\nChecking project folders...")
folders = [
    'models', 'uploads', 'heatmaps',
    'static', 'exports',
    'data/train/real', 'data/train/fake',
    'data/val/real',   'data/val/fake'
]
for folder in folders:
    exists = os.path.isdir(folder)
    status = '✅' if exists else '❌'
    if not exists:
        all_ok = False
    print(f"  {status} {folder}/")

# 5. Check files exist
print("\nChecking project files...")
files = [
    'app.py',
    'train.py',
    'static/dashboard.html',
    'requirements.txt',
]
for f in files:
    exists = os.path.isfile(f)
    status = '✅' if exists else '❌'
    if not exists:
        all_ok = False
    print(f"  {status} {f}")

# 6. Check training data
print("\nChecking training data...")
from pathlib import Path
for split in ['train', 'val']:
    for cls in ['real', 'fake']:
        p = Path(f'data/{split}/{cls}')
        if p.exists():
            imgs = (list(p.glob('*.jpg')) +
                    list(p.glob('*.jpeg')) +
                    list(p.glob('*.png')))
            status = '✅' if imgs else '⚠️ '
            print(f"  {status} data/{split}/{cls}: {len(imgs)} images")
        else:
            print(f"  ❌ data/{split}/{cls}: folder missing")

# 7. Check trained model
print("\nChecking trained model...")
if Path('models/truthlens_v4.pth').exists():
    size = Path('models/truthlens_v4.pth').stat().st_size / (1024*1024)
    print(f"  ✅ models/truthlens_v4.pth ({size:.1f} MB)")
else:
    print("  ⚠️  models/truthlens_v4.pth not found")
    print("     Run python train.py to create it")

# Final result
print("\n" + "="*50)
if all_ok:
    print("  ✅ All checks passed! Ready to run.")
    print("  Run: python app.py")
else:
    print("  ❌ Some checks failed — fix the issues above")
    print("  Run: pip install -r requirements.txt")
print("="*50 + "\n")