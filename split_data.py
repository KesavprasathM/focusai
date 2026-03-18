"""
TruthLens — Dataset Split Utility
Splits your images into train/val folders automatically.

Usage:
  python split_data.py --real C:/path/to/real_images --fake C:/path/to/fake_images
"""

import os
import shutil
import random
import argparse
from pathlib import Path

def split_folder(src_folder, train_dst, val_dst, val_ratio=0.2):
    src = Path(src_folder)
    if not src.exists():
        print(f"❌ Folder not found: {src}")
        return 0, 0

    # Find all images
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    images = []
    for ext in extensions:
        images += list(src.glob(f'*{ext}'))
        images += list(src.glob(f'*{ext.upper()}'))

    if not images:
        print(f"❌ No images found in {src}")
        return 0, 0

    random.shuffle(images)
    split     = int(len(images) * (1 - val_ratio))
    train_imgs = images[:split]
    val_imgs   = images[split:]

    # Create destination folders
    Path(train_dst).mkdir(parents=True, exist_ok=True)
    Path(val_dst).mkdir(parents=True, exist_ok=True)

    # Copy images
    print(f"Copying {len(train_imgs)} images to {train_dst}...")
    for f in train_imgs:
        shutil.copy(f, Path(train_dst) / f.name)

    print(f"Copying {len(val_imgs)} images to {val_dst}...")
    for f in val_imgs:
        shutil.copy(f, Path(val_dst) / f.name)

    return len(train_imgs), len(val_imgs)


def main():
    parser = argparse.ArgumentParser(description='Split images into train/val')
    parser.add_argument('--real', required=True,
                        help='Path to folder with REAL/authentic images')
    parser.add_argument('--fake', required=True,
                        help='Path to folder with FAKE/AI-generated images')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of images for validation (default 0.2)')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  TruthLens — Dataset Split Utility")
    print("="*50 + "\n")

    # Split real images
    print("📂 Processing REAL images...")
    tr, vr = split_folder(
        src_folder=args.real,
        train_dst='data/train/real',
        val_dst='data/val/real',
        val_ratio=args.val_ratio
    )

    # Split fake images
    print("\n📂 Processing FAKE images...")
    tf, vf = split_folder(
        src_folder=args.fake,
        train_dst='data/train/fake',
        val_dst='data/val/fake',
        val_ratio=args.val_ratio
    )

    # Summary
    print("\n" + "="*50)
    print("  Split Complete!")
    print(f"  data/train/real : {tr} images")
    print(f"  data/train/fake : {tf} images")
    print(f"  data/val/real   : {vr} images")
    print(f"  data/val/fake   : {vf} images")
    print(f"  Total           : {tr+tf+vr+vf} images")
    print("="*50)
    print("\n✅ Now run: python train.py --data ./data --epochs 20 --batch 8 --workers 0")


if __name__ == '__main__':
    main()