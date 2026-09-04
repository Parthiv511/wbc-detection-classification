# ============================================================
# BloodCell Intelligence
# ConvNeXt-Tiny WBC Classifier
# SINGLE BEST FOLD DEPLOYMENT
#
# Selected model: Fold 2
# Ensemble: disabled
# Classes: 13
# Device: CPU-friendly for Render
# ============================================================

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# DEVICE / CPU SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# WBCBENCH 13-CLASS MAPPING
# ============================================================

CLASS_NAMES: Dict[int, str] = {
    0: "BA",
    1: "BL",
    2: "BNE",
    3: "EO",
    4: "LY",
    5: "MMY",
    6: "MO",
    7: "MY",
    8: "PC",
    9: "PLY",
    10: "PMY",
    11: "SNE",
    12: "VLY",
}

CLASS_TO_ID: Dict[str, int] = {
    name: class_id
    for class_id, name in CLASS_NAMES.items()
}

NUM_CLASSES = 13
CONVNEXT_FEATURES = 768
IMAGE_SIZE = 224
CROP_PADDING = 0.15


# ============================================================
# RELIABILITY THRESHOLDS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.15
ENTROPY_THRESHOLD = 1.60


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = BASE_DIR / "weights"

SELECTED_FOLD = 2
CHECKPOINT_PATH = WEIGHTS_DIR / "convnext_wbc_fold2.pth"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# CREATE CONVNEXT-TINY
# ============================================================

def create_model() -> nn.Module:
    print("[Classifier] Creating ConvNeXt-Tiny model...")

    model = models.convnext_tiny(weights=None)

    classifier_layers = list(model.classifier.children())
    classifier_layers[-1] = nn.Linear(
        CONVNEXT_FEATURES,
        NUM_CLASSES,
    )

    model.classifier = nn.Sequential(*classifier_layers)
    model = model.to(DEVICE)
    model.eval()

    print("[Classifier] Model created.")
    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "ConvNeXt Fold 2 checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print(f"[Classifier] Loading best model: Fold {SELECTED_FOLD}")
    print(f"[Classifier] Checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError(
            "Unsupported ConvNeXt checkpoint format: "
            f"{type(checkpoint)}"
        )

    cleaned_state_dict: Dict[str, Any] = {}

    for key, value in state_dict.items():
        clean_key = key

        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]

        cleaned_state_dict[clean_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing_keys:
        raise RuntimeError(
            "Missing ConvNeXt Fold 2 checkpoint keys:\n"
            + "\n".join(str(key) for key in missing_keys)
        )

    if unexpected_keys:
        print("[Classifier] Warning: unexpected checkpoint keys:")
        for key in unexpected_keys:
            print(f"  {key}")

    del cleaned_state_dict
    del state_dict
    del checkpoint
    gc.collect()

    model.eval()

    print("[Classifier] Fold 2 loaded successfully.")
    return model


# ============================================================
# SINGLE MODEL INSTANCE
# ============================================================

MODEL: nn.Module | None = None
MODEL_READY = False


def get_model() -> nn.Module:
    """Return the single loaded ConvNeXt Fold 2 model."""

    global MODEL, MODEL_READY

    if MODEL is None:
        MODEL = create_model()
        load_checkpoint(MODEL, CHECKPOINT_PATH)
        MODEL_READY = True
        print("[Classifier] Fold 2 model initialization successful.")

    return MODEL


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = TRANSFORM(image)

    # Explicit tensor conversion avoids static-analysis issues
    # around PIL/Image-like objects and unsqueeze.
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    return tensor


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(probabilities: torch.Tensor) -> float:
    probabilities = torch.clamp(
        probabilities,
        min=1e-8,
        max=1.0,
    )

    entropy = -torch.sum(
        probabilities * torch.log(probabilities)
    )

    return float(entropy.item())


# ============================================================
# TOP-2 MARGIN
# ============================================================

def calculate_margin(probabilities: torch.Tensor) -> float:
    sorted_probabilities = torch.sort(
        probabilities,
        descending=True,
    ).values

    if sorted_probabilities.numel() < 2:
        return 0.0

    return float(
        (
            sorted_probabilities[0]
            - sorted_probabilities[1]
        ).item()
    )


# ============================================================
# RELIABILITY
# ============================================================

def evaluate_reliability(
    confidence: float,
    margin: float,
    entropy: float,
) -> Dict[str, Any]:
    if (
        confidence >= 0.70
        and margin >= 0.20
        and entropy <= 1.20
    ):
        return {
            "status": "high",
            "reliable": True,
            "reason": (
                "Strong confidence and low prediction uncertainty "
                "from the selected Fold 2 model."
            ),
        }

    if (
        confidence >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
        and entropy <= ENTROPY_THRESHOLD
    ):
        return {
            "status": "moderate",
            "reliable": True,
            "reason": (
                "Acceptable confidence and prediction separation "
                "from the selected Fold 2 model."
            ),
        }

    return {
        "status": "low",
        "reliable": False,
        "reason": (
            "Prediction has lower confidence or higher uncertainty."
        ),
    }


# ============================================================
# CLASSIFY ONE WBC CROP
# ============================================================

@torch.inference_mode()
def classify_wbc_image(image: Image.Image) -> Dict[str, Any]:
    model = get_model()
    tensor = prepare_image(image)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    probabilities = probabilities.detach().cpu()

    confidence_tensor, class_tensor = torch.max(
        probabilities,
        dim=0,
    )

    predicted_class = int(class_tensor.item())
    confidence = float(confidence_tensor.item())

    class_name = CLASS_NAMES.get(
        predicted_class,
        "UNKNOWN",
    )

    margin = calculate_margin(probabilities)
    entropy = calculate_entropy(probabilities)

    # Only one model/fold is used, therefore agreement is 1.0.
    fold_agreement = 1.0

    reliability = evaluate_reliability(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
    )

    probability_values = probabilities.tolist()
    probabilities_dict: Dict[str, float] = {}

    for class_id, probability in enumerate(probability_values):
        probabilities_dict[CLASS_NAMES[class_id]] = round(
            float(probability),
            6,
        )

    result: Dict[str, Any] = {
        "class_id": predicted_class,
        "class_name": class_name,
        "subtype": class_name,
        "raw_class_name": class_name,
        "final_decision": class_name,
        "confidence": round(confidence, 6),
        "margin": round(margin, 6),
        "entropy": round(entropy, 6),
        "fold_agreement": fold_agreement,
        "reliability": reliability["status"],
        "reliable": reliability["reliable"],
        "reliability_reason": reliability["reason"],
        "decision_method": "single_fold",
        "selected_fold": SELECTED_FOLD,
        "votes": 1,
        "vote_count": 1,
        "vote_total": 1,
        "majority_vote": False,
        "fold_predictions": [
            {
                "fold": SELECTED_FOLD,
                "class_id": predicted_class,
                "class_name": class_name,
                "confidence": round(confidence, 6),
            }
        ],
        "probabilities": probabilities_dict,
        "model": "ConvNeXt-Tiny",
        "ensemble": "disabled",
        "num_classes": NUM_CLASSES,
        "input_size": IMAGE_SIZE,
        "available_folds": [SELECTED_FOLD],
    }

    del tensor
    del logits
    del probabilities
    del confidence_tensor
    del class_tensor
    gc.collect()

    return result


# ============================================================
# CROP WBC FROM YOLO BOUNDING BOX
# ============================================================

def crop_wbc(
    image: Image.Image,
    bbox: Dict[str, float],
    padding: float = CROP_PADDING,
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_width, image_height = image.size

    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])

    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid WBC bounding box.")

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = box_width * padding
    pad_y = box_height * padding

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(image_width, int(x2 + pad_x))
    crop_y2 = min(image_height, int(y2 + pad_y))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("WBC crop has invalid dimensions.")

    crop = image.crop(
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )
    )

    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    return crop


# ============================================================
# CLASSIFY WBC USING YOLO BBOX
# ============================================================

def classify_wbc_crop(
    image: Image.Image,
    bbox: Dict[str, float],
) -> Dict[str, Any]:
    crop = crop_wbc(
        image=image,
        bbox=bbox,
        padding=CROP_PADDING,
    )

    result = classify_wbc_image(crop)

    result["crop_padding"] = CROP_PADDING
    result["crop_strategy"] = "padding_15"
    result["input_size"] = [IMAGE_SIZE, IMAGE_SIZE]

    return result


# ============================================================
# SIMPLE PREDICT FUNCTION
# ============================================================

def predict(image: Image.Image) -> Dict[str, Any]:
    return classify_wbc_image(image)


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict[str, Any]:
    return {
        "model": "ConvNeXt-Tiny",
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "selected_fold": SELECTED_FOLD,
        "available_folds": [SELECTED_FOLD],
        "ensemble": "disabled",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "input_size": IMAGE_SIZE,
        "feature_dimension": CONVNEXT_FEATURES,
        "task": "WBC subtype classification",
        "decision_method": "single_fold",
        "crop_strategy": "padding_15",
        "crop_padding": CROP_PADDING,
        "device": str(DEVICE),
    }


# ============================================================
# STARTUP TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print("CONVNEXT CLASSIFIER READY")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print("Model: ConvNeXt-Tiny")
    print("Selected model: Fold 2")
    print("Mode: SINGLE BEST FOLD")
    print("Ensemble: DISABLED")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Input size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("Available folds: [2]")
    print("=" * 64)

    try:
        get_model()
        print("Fold 2 model initialization successful.")
    except Exception as exc:
        print("Fold 2 model initialization failed:")
        print(exc)
        raise
