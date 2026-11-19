"""Inference helper matching image-based-eye-diseases-detection-final.ipynb.

This is a research prototype, not a medical device or diagnostic system. Scores
are not clinical probabilities; internal temperature scaling is not clinical
calibration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CLASS_NAMES = ["cataract", "glaucoma", "normal"]
MODEL_NAME = "ResNet50 with custom 3-class classifier head"
PREPROCESSING = {
    "input_color": "RGB",
    "resize": [224, 224],
    "tensor": "ToTensor (float32, [0,1])",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
}


class EyeClassificationModel(nn.Module):
    """Architecture reconstructed from the notebook; state-dict compatible."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.feature_extractor = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_transform() -> transforms.Compose:
    """Use the notebook's deterministic evaluation transform, not train augments."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class EyeDiseasePredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = EyeClassificationModel(num_classes=len(CLASS_NAMES))
        # weights_only=True avoids unpickling arbitrary Python objects on supported PyTorch.
        try:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility for older PyTorch releases
            state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError("Expected a state_dict or a checkpoint containing 'state_dict'.")
        # Deliberately strict: shape/key mismatch should stop rather than silently run.
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.transform = build_transform()
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        calibration_report = self.checkpoint_path.parent / "fresh_group_split_metrics.json"
        self.training_evaluation: dict[str, Any] | None = None
        if temperature is not None:
            self.temperature = float(temperature)
            self.calibration = {
                "status": "temperature supplied by caller",
                "method": "temperature scaling",
                "temperature": self.temperature,
            }
        elif calibration_report.is_file():
            report = json.loads(calibration_report.read_text(encoding="utf-8"))
            if report.get("checkpoint_sha256") == self.checkpoint_sha256:
                self.temperature = float(report["calibration"]["temperature"])
                def summary(metrics: dict[str, Any]) -> dict[str, float]:
                    fields = ("accuracy", "negative_log_likelihood", "multiclass_brier_score", "top_label_ece_15_bins")
                    return {field: float(metrics[field]) for field in fields if field in metrics}

                self.training_evaluation = {
                    "experiment": report.get("experiment"),
                    "dataset_policy": report.get("dataset_policy"),
                    "split_policy": report.get("split_policy"),
                    "group_policy": report.get("group_policy"),
                    "split_sizes": report.get("split_sizes"),
                    "training_summary": {
                        key: report.get("training", {}).get(key)
                        for key in ("epochs_requested", "epochs_completed", "best_validation_accuracy")
                    },
                    "calibration_partition_raw_metrics": summary(report["calibration"]["calibration_raw"]),
                    "calibration_partition_temperature_scaled_metrics": summary(report["calibration"]["calibration_scaled"]),
                    "final_test_raw_metrics": summary(report["final_test_raw"]),
                    "final_test_temperature_scaled_metrics": summary(report["final_test_temperature_scaled"]),
                    "interpretation": "Aggregate internal-dataset results only; not external or clinical validation.",
                }
                self.calibration = {
                    "status": "internally calibrated on a reserved dataset partition",
                    "method": report["calibration"]["method"],
                    "temperature": self.temperature,
                    "report_file": calibration_report.name,
                    "evaluation_note": "Internal split only; not clinical calibration.",
                }
            else:
                self.temperature = 1.0
                self.calibration = {
                    "status": "calibration metadata did not match this checkpoint; raw softmax only",
                    "method": None,
                    "temperature": 1.0,
                }
        else:
            self.temperature = 1.0
            self.calibration = {
                "status": "not established; raw softmax only",
                "method": None,
                "temperature": 1.0,
            }
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Return every class's raw and temperature-scaled score plus provenance."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)[0]
        raw_scores = torch.softmax(logits, dim=0).detach().cpu().tolist()
        adjusted_scores = torch.softmax(logits / self.temperature, dim=0).detach().cpu().tolist()
        ranked = sorted(
            ({"class": name, "score": float(score)} for name, score in zip(CLASS_NAMES, adjusted_scores)),
            key=lambda item: item["score"], reverse=True,
        )
        raw_ranked = sorted(
            ({"class": name, "score": float(score)} for name, score in zip(CLASS_NAMES, raw_scores)),
            key=lambda item: item["score"], reverse=True,
        )
        return {
            "predictions": ranked,
            "raw_softmax_scores": raw_ranked,
            "top_prediction": ranked[0]["class"],
            "top_score": ranked[0]["score"],
            "score_semantics": "Temperature-scaled model scores; empirical calibration only on the supplied internal dataset, not clinical probabilities.",
            "calibration": self.calibration,
            "metadata": {
                "model": MODEL_NAME,
                "classes_in_logit_order": CLASS_NAMES,
                "preprocessing": PREPROCESSING,
                "checkpoint_file": self.checkpoint_path.name,
                "checkpoint_sha256": self.checkpoint_sha256,
                "device": str(self.device),
                "input_image_size_px": list(image.size),
                "training_evaluation": self.training_evaluation,
            },
            "limitations": [
                "Research prototype; not a confirmed diagnosis or substitute for clinician review.",
                "Only cataract, glaucoma, and normal are represented by this classifier.",
                "Temperature scaling improved aggregate internal-test NLL and ECE, but does not make these scores clinical probabilities.",
                "Performance may differ for images, devices, populations, or settings outside the evaluation data.",
            ],
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        with Image.open(image_path) as image:
            return self.predict(image)
