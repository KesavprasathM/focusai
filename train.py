"""
TruthLens v4.0 — Training Script (NaN Fixed)
Run:
  python train.py --data ./data --epochs 20 --batch 8 --lr 5e-5 --workers 0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import v2 as T
from sklearn.metrics import roc_auc_score, f1_score
import sys
sys.path.insert(0, '.')
from app import TruthLensV4


# ── AUGMENTATION
def get_transforms(split='train', img_size=380):
    if split == 'train':
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                         saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
        ])


# ── LABEL SMOOTHING LOSS
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        smooth_targets = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = (smooth_targets * (1 - self.smoothing)
                         + self.smoothing / n_classes)
        log_probs = torch.log_softmax(logits, dim=1)
        return -(smooth_targets * log_probs).sum(dim=1).mean()


# ── TRAINING LOOP (NaN Safe)
def train_epoch(model, loader, optimiser, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        with torch.autocast(device_type=device.type,
                            enabled=(device.type == 'cuda')):
            logits = model(images)
            loss   = criterion(logits, labels)

        # ── Skip batch if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            optimiser.zero_grad()
            if nan_batches > 10:
                print(f"  ⚠️  Too many NaN batches ({nan_batches}), stopping epoch early")
                break
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)

        # ── Tighter gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        # ── Skip step if gradients contain NaN
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if has_nan_grad:
            nan_batches += 1
            optimiser.zero_grad()
            scaler.update()
            continue

        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        preds    = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if i % 20 == 0:
            print(f"  Batch [{i:4d}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  "
                  f"acc={correct/max(total,1):.3f}")

    if nan_batches > 0:
        print(f"  ⚠️  Skipped {nan_batches} NaN batches this epoch")

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ── VALIDATION LOOP
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs  = []
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, 1)[:, 1]
        probs  = torch.clamp(probs, 1e-7, 1 - 1e-7)
        preds  = logits.argmax(1)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        all_probs.extend(probs.cpu().float().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_probs  = np.nan_to_num(np.array(all_probs,  dtype=np.float32), nan=0.5)
    all_preds  = np.array(all_preds,  dtype=np.int32)
    all_labels = np.nan_to_num(np.array(all_labels, dtype=np.float32), nan=0.0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    try:
        f1 = f1_score(all_labels, all_preds, average='weighted')
    except Exception:
        f1 = 0.0

    return total_loss / max(len(loader), 1), correct / max(total, 1), auc, f1


# ── MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='./data')
    parser.add_argument('--epochs',  type=int,   default=20)
    parser.add_argument('--batch',   type=int,   default=8)
    parser.add_argument('--lr',      type=float, default=5e-5)
    parser.add_argument('--workers', type=int,   default=0)
    parser.add_argument('--resume',  default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*55)
    print("  TruthLens v4.0 — Training Script (NaN Fixed)")
    print("="*55)
    print(f"  Device  : {device}")
    if device.type == 'cuda':
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print("="*55 + "\n")

    train_path = Path(args.data) / 'train'
    val_path   = Path(args.data) / 'val'

    if not train_path.exists():
        print(f"ERROR: {train_path} not found!")
        return

    train_ds = datasets.ImageFolder(train_path, transform=get_transforms('train'))
    val_ds   = datasets.ImageFolder(val_path,   transform=get_transforms('val'))

    print(f"Train : {len(train_ds):,} images")
    print(f"Val   : {len(val_ds):,} images")
    print(f"Classes: {train_ds.class_to_idx}\n")

    counts  = np.bincount(train_ds.targets)
    weights = 1.0 / counts[train_ds.targets]
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    model = TruthLensV4(num_classes=2).to(device)

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"✅ Resumed from: {args.resume}\n")

    # ── Phase 1: Freeze backbone (epochs 1-5)
    print("Phase 1: Backbone frozen (epochs 1-5)")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimiser = AdamW([
        {'params': model.freq_branch.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(),  'lr': args.lr},
        {'params': model.backbone.parameters(),    'lr': args.lr * 0.01},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2)
    criterion = LabelSmoothingCE(smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    Path('models').mkdir(exist_ok=True)
    save_path = Path('models/truthlens_v4.pth')
    best_auc  = 0.0

    for epoch in range(1, args.epochs + 1):

        # ── Phase 2: Unfreeze backbone at epoch 6 with very low LR
        if epoch == 6:
            print("\nPhase 2: Backbone unfrozen with low LR!")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Very conservative LRs to prevent NaN explosion
            optimiser.param_groups[0]['lr'] = args.lr * 0.1
            optimiser.param_groups[1]['lr'] = args.lr * 0.1
            optimiser.param_groups[2]['lr'] = args.lr * 0.01

        print(f"\n--- Epoch {epoch:02d}/{args.epochs} ---")
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimiser, criterion, device, scaler
        )
        vl_loss, vl_acc, vl_auc, vl_f1 = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:02d} Summary:")
        print(f"  Train — loss:{tr_loss:.4f}  acc:{tr_acc:.3f}")
        print(f"  Val   — loss:{vl_loss:.4f}  acc:{vl_acc:.3f}")
        print(f"  AUC   — {vl_auc:.4f}   F1:{vl_f1:.4f}")
        print(f"  Time  — {elapsed:.1f}s")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ BEST MODEL SAVED! AUC={best_auc:.4f} → {save_path}")

    print("\n" + "="*55)
    print("  TRAINING COMPLETE!")
    print(f"  Best AUC : {best_auc:.4f}")
    print(f"  Model    : {save_path}")
    print("="*55)
    print("\nNext: python app.py")


if __name__ == '__main__':
    main()
    #python train.py --data ./data --epochs 20 --batch 8 --lr 5e-5 --workers 0 --resume models/truthlens_v4.pth to resume training