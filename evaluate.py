"""Exploratory evaluation and post-hoc calibration for the eye CAD model.

Creates a deterministic stratified 60/20/20 resplit of current JPG files and
fits temperature scaling on the resplit's validation data. It does NOT know
the notebook's original path ordering or saved split assignments. If the
checkpoint was trained/selected on this source dataset, resulting metrics may
be contaminated by overlap and must not be treated as independent test results.
No images, paths, or per-image predictions are written to the output JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from inference import CLASS_NAMES, EyeClassificationModel


class ImageDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label = self.entries[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            return self.transform(im), label


def make_splits(root: Path):
    # Match notebook: only immediate class folders, .jpg images, alphabetical classes.
    class_folders = {p.name: p for p in root.iterdir() if p.is_dir()}
    missing = set(CLASS_NAMES) - set(class_folders)
    if missing:
        raise ValueError(f"Missing required class folders: {sorted(missing)}")
    entries, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        paths = sorted(class_folders[name].glob("*.jpg"))
        entries.extend((p, label) for p in paths)
        labels.extend([label] * len(paths))
    if not entries:
        raise ValueError(f"No lowercase .jpg images found directly inside {root}/<class>/")
    trv, test, y_trv, y_test = train_test_split(
        entries, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train, val, y_train, y_val = train_test_split(
        trv, y_trv, test_size=0.25, random_state=42, stratify=y_trv
    )
    return train, val, test


def collect_logits(model, entries, transform, device, batch_size):
    loader = DataLoader(ImageDataset(entries, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    logits, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            labels.extend(y.numpy().tolist())
    return torch.cat(logits), np.asarray(labels, dtype=int)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.any():
            ece += mask.sum() / total * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def metrics(logits: torch.Tensor, y: np.ndarray, temperature: float) -> dict:
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "nll": float(log_loss(y, probs, labels=np.arange(len(CLASS_NAMES)))),
        "multiclass_brier_score": float(np.mean(np.sum((probs - np.eye(len(CLASS_NAMES))[y]) ** 2, axis=1))),
        "top_label_ece_15_bins": expected_calibration_error(probs, y),
        "confusion_matrix_rows_true_cols_predicted": confusion_matrix(
            y, pred, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "classification_report": classification_report(
            y, pred, labels=np.arange(len(CLASS_NAMES)), target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset root with cataract/glaucoma/normal folders")
    parser.add_argument("--checkpoint", default="best_eye_disease_model.pth")
    parser.add_argument("--output", default="evaluation_metrics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory unavailable: {root}")
    train, val, test = make_splits(root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = EyeClassificationModel(num_classes=len(CLASS_NAMES))
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_logits, y_val = collect_logits(model, val, transform, device, args.batch_size)
    test_logits, y_test = collect_logits(model, test, transform, device, args.batch_size)

    # Optimize temperature on validation NLL only; test labels are not used for fitting.
    # A sigmoid parameterization constrains the search to [0.05, 20].
    raw_t = torch.tensor(-2.995732, requires_grad=True)  # corresponds approximately to T=1
    optimizer = torch.optim.LBFGS([raw_t], lr=0.1, max_iter=100)
    val_targets = torch.as_tensor(y_val, dtype=torch.long)
    def closure():
        optimizer.zero_grad()
        temperature = 0.05 + 19.95 * torch.sigmoid(raw_t)
        loss = torch.nn.functional.cross_entropy(val_logits / temperature, val_targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float((0.05 + 19.95 * torch.sigmoid(raw_t.detach())).item())

    result = {
        "evaluation_protocol": "Exploratory deterministic re-split of current JPG files (lexicographically sorted paths); stratified 60/20/20, random_state=42; temperature fitted on this re-split's validation only. This is not verified to match the notebook's original sample assignments.",
        "classes_in_logit_order": CLASS_NAMES,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "calibration": {"method": "single scalar temperature scaling", "temperature_fit_on_validation": temperature,
                        "validation_raw": metrics(val_logits, y_val, 1.0),
                        "validation_temperature_scaled": metrics(val_logits, y_val, temperature)},
        "test_raw": metrics(test_logits, y_test, 1.0),
        "test_temperature_scaled": metrics(test_logits, y_test, temperature),
        "caveats": [
            "Temperature scaling was fit on the validation split; test metrics are held out from that fit.",
            "The checkpoint was trained/selected using this dataset; without original split IDs, the re-created validation and test splits may overlap samples used for training or checkpoint selection. Treat all results as exploratory, not independent test or calibration evidence.",
            "Re-split test accuracy did not match the notebook's recorded 88.53%; this confirms that the original sample assignments were not reproduced.",
            "The split is image-level, as in the notebook; patient-level independence cannot be verified from the notebook/data layout.",
            "This evaluation is not external or clinical validation and does not establish clinical utility.",
            "Only aggregate metrics are written; image paths and per-image outputs are excluded.",
        ],
    }
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "split_sizes": result["split_sizes"],
                      "temperature": temperature, "test_raw_accuracy": result["test_raw"]["accuracy"],
                      "test_calibrated_accuracy": result["test_temperature_scaled"]["accuracy"],
                      "test_raw_nll": result["test_raw"]["nll"],
                      "test_calibrated_nll": result["test_temperature_scaled"]["nll"]}, indent=2))


if __name__ == "__main__":
    main()
