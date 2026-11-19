"""Inference helper matching image-based-eye-diseases-detection-final.ipynb.

This is a research prototype, not a medical device or diagnostic system. The
softmax outputs are model scores; they have not been shown to be calibrated.
"""
from __future__ import annotations

import hashlib
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
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
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

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Return all class softmax scores and traceable model metadata."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)[0]
        scores = torch.softmax(logits, dim=0).detach().cpu().tolist()
        ranked = sorted(
            ({"class": name, "score": float(score)} for name, score in zip(CLASS_NAMES, scores)),
            key=lambda item: item["score"], reverse=True,
        )
        return {
            "predictions": ranked,
            "top_prediction": ranked[0]["class"],
            "top_score": ranked[0]["score"],
            "score_semantics": "softmax model score; calibration has not been established",
            "metadata": {
                "model": MODEL_NAME,
                "classes_in_logit_order": CLASS_NAMES,
                "preprocessing": PREPROCESSING,
                "checkpoint_file": self.checkpoint_path.name,
                "checkpoint_sha256": self.checkpoint_sha256,
                "device": str(self.device),
                "input_image_size_px": list(image.size),
            },
            "limitations": [
                "Research prototype; not a confirmed diagnosis or substitute for clinician review.",
                "Only cataract, glaucoma, and normal are represented by this classifier.",
                "Softmax scores are not validated or calibrated clinical probabilities.",
                "Performance may differ for images, devices, populations, or settings outside the evaluation data.",
            ],
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        with Image.open(image_path) as image:
            return self.predict(image)
