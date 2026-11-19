"""Fresh, group-aware training/evaluation for the three-class eye CAD prototype.

This deliberately does NOT initialize from the supplied CAD checkpoint because
its original split assignments are unavailable. It starts from ImageNet weights,
keeps paired left/right IDs in one partition, uses separate validation and
calibration partitions, and evaluates the test partition only after training.
Only JPGs are included to match the original notebook's data loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["cataract", "glaucoma", "normal"]


class FundusDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, _group = self.entries[idx]
        with Image.open(path) as image:
            # Same as notebook: PIL decode then RGB; do not apply EXIF orientation.
            image = image.convert("RGB")
            return self.transform(image), label


class EyeClassificationModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Fresh ImageNet initialization only; never load the existing CAD checkpoint.
        self.feature_extractor = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False
        feature_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.feature_extractor(x)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def discover_entries(root: Path):
    entries = []
    for label, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.glob("*.jpg")):
            stem = path.stem
            # Eye-image filenames use IDs ending in _left/_right. Keep both eyes
            # and any same-ID cross-class records in the same split conservatively.
            group = stem.rsplit("_", 1)[0] if stem.endswith(("_left", "_right")) else stem
            entries.append((path, label, group))
    if not entries:
        raise ValueError(f"No .jpg files found under {root}/<class>/")
    return entries


def make_group_splits(entries):
    y = np.asarray([e[1] for e in entries], dtype=int)
    groups = np.asarray([e[2] for e in entries])
    x = np.arange(len(entries))
    # One fifth (approximately 20%) of patient/ID groups is reserved as final test.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=421)
    trainval_idx, test_idx = next(outer.split(x, y, groups))
    # Remaining 80% is split into 8 approximately equal folds: one validation,
    # one calibration, and six training folds -> about 10/10/60% of all images.
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=422)
    inner_folds = list(inner.split(x[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    val_idx = trainval_idx[inner_folds[0][1]]
    calibration_idx = trainval_idx[inner_folds[1][1]]
    reserved = set(val_idx.tolist()) | set(calibration_idx.tolist())
    train_idx = np.asarray([i for i in trainval_idx if i not in reserved], dtype=int)
    split_indices = {"train": train_idx, "validation": val_idx,
                     "calibration": calibration_idx, "test": test_idx}
    group_sets = {name: set(groups[idx]) for name, idx in split_indices.items()}
    for i, a in enumerate(split_indices):
        for b in list(split_indices)[i + 1:]:
            overlap = group_sets[a] & group_sets[b]
            if overlap:
                raise RuntimeError(f"Group leakage between {a} and {b}: {len(overlap)} IDs")
    for name, idx in split_indices.items():
        if set(y[idx]) != set(range(len(CLASSES))):
            raise RuntimeError(f"Split {name} does not contain all classes")
    return {name: [entries[i] for i in idx] for name, idx in split_indices.items()}


def load_logits(model, entries, transform, device, batch_size):
    loader = DataLoader(FundusDataset(entries, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            all_logits.append(model(images.to(device)).cpu())
            all_labels.extend(labels.tolist())
            if batch_idx % 20 == 0 or batch_idx == len(loader):
                print(f"  inference batch {batch_idx}/{len(loader)}", flush=True)
    return torch.cat(all_logits), np.asarray(all_labels, dtype=int)


def ece_score(probs, y, bins=15):
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (confidence > edges[b]) & (confidence <= edges[b + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def summarize(logits, y, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(CLASSES)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "negative_log_likelihood": float(log_loss(y, probs, labels=np.arange(len(CLASSES)))),
        "multiclass_brier_score": float(np.mean(np.sum((probs - np.eye(len(CLASSES))[y]) ** 2, axis=1))),
        "top_label_ece_15_bins": ece_score(probs, y),
        "confusion_matrix_rows_true_cols_predicted": confusion_matrix(
            y, pred, labels=np.arange(len(CLASSES))
        ).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
        "classification_report": classification_report(
            y, pred, labels=np.arange(len(CLASSES)), target_names=CLASSES,
            output_dict=True, zero_division=0
        ),
    }


def fit_temperature(logits, labels):
    raw_t = torch.tensor(-2.995732, dtype=torch.float32, requires_grad=True)
    target = torch.as_tensor(labels, dtype=torch.long)
    optimizer = torch.optim.LBFGS([raw_t], lr=0.1, max_iter=100)
    def closure():
        optimizer.zero_grad()
        temperature = 0.05 + 19.95 * torch.sigmoid(raw_t)
        loss = nn.functional.cross_entropy(logits / temperature, target)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float((0.05 + 19.95 * torch.sigmoid(raw_t.detach())).item())


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="fresh_group_split_run")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    root = Path(args.dataset).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = discover_entries(root)
    splits = make_group_splits(entries)
    split_sizes, split_counts, split_groups = {}, {}, {}
    for name, items in splits.items():
        split_sizes[name] = len(items)
        split_counts[name] = {c: sum(1 for _, y, _ in items if y == i)
                              for i, c in enumerate(CLASSES)}
        split_groups[name] = len({g for _, _, g in items})
    print("Group-aware split counts:", json.dumps(split_sizes), flush=True)
    print("Class counts:", json.dumps(split_counts), flush=True)
    print("Unique group counts:", json.dumps(split_groups), flush=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(FundusDataset(splits["train"], train_transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FundusDataset(splits["validation"], eval_transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = EyeClassificationModel(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_accuracy = -1.0
    history = []
    checkpoint = out / "fresh_group_split_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch_no, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += len(labels)
            if batch_no % 25 == 0 or batch_no == len(train_loader):
                print(f"epoch={epoch}/{args.epochs} train_batch={batch_no}/{len(train_loader)}", flush=True)

        model.eval()
        val_loss_sum, val_correct, val_seen = 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss_sum += float(criterion(logits, labels).item()) * len(labels)
                val_correct += int((logits.argmax(1) == labels).sum().item())
                val_seen += len(labels)
        val_loss, val_acc = val_loss_sum / val_seen, val_correct / val_seen
        scheduler.step(val_loss)
        epoch_record = {
            "epoch": epoch, "train_loss": total_loss / seen,
            "train_accuracy": correct / seen, "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint)
            print(f"Saved best checkpoint (validation accuracy={val_acc:.4f})", flush=True)

    # Calibration is fit exclusively on its reserved split; final test remains untouched until here.
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    calibration_logits, y_cal = load_logits(model, splits["calibration"], eval_transform, device, args.batch_size)
    temperature = fit_temperature(calibration_logits, y_cal)
    test_logits, y_test = load_logits(model, splits["test"], eval_transform, device, args.batch_size)

    report = {
        "experiment": "fresh ImageNet-initialized ResNet50; no initialization from prior CAD checkpoint",
        "classes_in_logit_order": CLASSES,
        "dataset_policy": "Only .jpg files; RGB conversion and notebook preprocessing; uploaded PNGs excluded.",
        "split_policy": "Group-aware image-ID split: approximately 60% train, 10% model-selection validation, 10% temperature calibration, 20% final test.",
        "group_policy": "Filename stem with _left/_right suffix removed; same identifier kept within one split across class folders.",
        "split_sizes": split_sizes,
        "class_counts": split_counts,
        "unique_group_counts": split_groups,
        "group_overlap_checks": "All pairwise group overlaps were asserted to be zero.",
        "training": {"epochs_requested": args.epochs, "epochs_completed": len(history),
                     "best_validation_accuracy": best_val_accuracy, "history": history},
        "calibration": {"method": "single scalar temperature scaling", "temperature": temperature,
                        "calibration_raw": summarize(calibration_logits, y_cal, 1.0),
                        "calibration_scaled": summarize(calibration_logits, y_cal, temperature)},
        "final_test_raw": summarize(test_logits, y_test, 1.0),
        "final_test_temperature_scaled": summarize(test_logits, y_test, temperature),
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "limitations": [
            "Internal evaluation on a single supplied dataset; no external or clinical validation.",
            "Filename-derived IDs are treated as subject groups because filenames have left/right eye suffixes; source semantics should be confirmed by dataset documentation.",
            "Calibration is fit on one finite calibration partition; calibration uncertainty remains and scores are not clinical probabilities.",
            "No threshold for clinical decision-making is established; clinician review is required.",
        ],
    }
    report_path = out / "fresh_group_split_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TRAINING COMPLETE", flush=True)
    print(json.dumps({
        "checkpoint": str(checkpoint), "report": str(report_path),
        "temperature": temperature,
        "test_raw_accuracy": report["final_test_raw"]["accuracy"],
        "test_calibrated_accuracy": report["final_test_temperature_scaled"]["accuracy"],
        "test_raw_nll": report["final_test_raw"]["negative_log_likelihood"],
        "test_calibrated_nll": report["final_test_temperature_scaled"]["negative_log_likelihood"],
        "test_raw_ece": report["final_test_raw"]["top_label_ece_15_bins"],
        "test_calibrated_ece": report["final_test_temperature_scaled"]["top_label_ece_15_bins"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
