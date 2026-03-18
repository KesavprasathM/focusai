"""
TruthLens v4.0 — Model Evaluation Script
Tests your trained model accuracy on validation data.
Shows accuracy, AUC, confusion matrix, and per-class stats.

Usage:
  python evaluate_model.py --data ./data
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)
import sys
sys.path.insert(0, '.')
from app import TruthLensV4

def evaluate(data_path, batch_size=16):
    print("\n" + "="*50)
    print("  TruthLens v4.0 — Model Evaluation")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    # Load model
    model_path = Path('models/truthlens_v4.pth')
    if not model_path.exists():
        print("❌ No trained model found!")
        print("   Run python train.py first")
        return

    model = TruthLensV4(num_classes=2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded\n")

    # Val dataset
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_path = Path(data_path) / 'val'
    if not val_path.exists():
        print(f"❌ Val folder not found at {val_path}")
        return

    val_ds = datasets.ImageFolder(val_path, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    print(f"Val samples : {len(val_ds)}")
    print(f"Classes     : {val_ds.class_to_idx}")
    print(f"\nRunning evaluation...")

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, 1)[:, 1]
            preds  = logits.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

            if i % 10 == 0:
                print(f"  Batch {i}/{len(val_loader)}...")

    # Metrics
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    auc      = roc_auc_score(all_labels, all_probs)
    cm       = confusion_matrix(all_labels, all_preds)
    report   = classification_report(
        all_labels, all_preds,
        target_names=['Fake/AI', 'Real/Authentic']
    )

    # Find best threshold
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_thresh, best_acc = 0.5, 0.0
    for t in thresholds:
        preds_t = (all_probs >= t).astype(int)
        acc_t   = accuracy_score(all_labels, preds_t)
        if acc_t > best_acc:
            best_acc    = acc_t
            best_thresh = t

    # Print results
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"\n  Accuracy : {accuracy*100:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print(f"\n  Best Threshold : {best_thresh:.2f} "
          f"(Accuracy: {best_acc*100:.2f}%)")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':10} Pred:Fake  Pred:Real")
    print(f"  {'True:Fake':10} {cm[0][0]:9}  {cm[0][1]:9}")
    print(f"  {'True:Real':10} {cm[1][0]:9}  {cm[1][1]:9}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (Real  detected correctly) : {tp}")
    print(f"  True Negatives  (Fake  detected correctly) : {tn}")
    print(f"  False Positives (Real  wrongly flagged AI) : {fp}")
    print(f"  False Negatives (Fake  missed)             : {fn}")

    print(f"\n  Per-Class Report:")
    print(report)

    # Grade
    grade = (
        "🏆 Excellent" if auc >= 0.95 else
        "✅ Good"      if auc >= 0.90 else
        "⚠️  Fair"     if auc >= 0.80 else
        "❌ Poor — needs more training data"
    )
    print(f"  Overall Grade : {grade}")
    print("="*50 + "\n")

    # Save report
    report_path = Path('exports/evaluation_report.txt')
    Path('exports').mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"TruthLens v4.0 Evaluation Report\n")
        f.write(f"{'='*40}\n")
        f.write(f"Accuracy : {accuracy*100:.2f}%\n")
        f.write(f"AUC      : {auc:.4f}\n")
        f.write(f"Grade    : {grade}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{report}\n")
    print(f"  Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default='./data')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    evaluate(args.data, args.batch)


if __name__ == '__main__':
    main()