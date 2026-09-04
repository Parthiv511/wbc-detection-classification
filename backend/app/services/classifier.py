# ============================================================
# BloodCell Intelligence
# WBCBench ConvNeXt-Tiny Classifier
#
# PRODUCTION VERSION
# ------------------------------------------------------------
# Uses ONLY Fold 2 because Fold 2 achieved the best evaluation
# performance.
#
# MEMORY OPTIMIZED FOR RENDER / LOW-RAM DEPLOYMENT
# ============================================================

from pathlib import Path
from typing import Any, Dict

import gc

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# DEVICE
# ============================================================

# Render free tier has limited RAM.
# CPU inference is intentionally used.
DEVICE = torch.device("cpu")


# Reduce CPU/thread overhead on small Render instances.
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


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 13

# ConvNeXt-Tiny final feature dimension.
CONVNEXT_FEATURES = 768

IMAGE_SIZE = 224

# Extra area around YOLO bounding box.
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

# backend/
#   app/
#      services/
#          classifier.py
#
# parents[2] -> backend
#
BASE_DIR = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = BASE_DIR / "weights"


# ============================================================
# IMPORTANT:
# ONLY FOLD 2 IS USED
# ============================================================

FOLD_2_PATH = (
    WEIGHTS_DIR / "convnext_wbc_fold2.pth"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model() -> nn.Module:
    """
    Create an uninitialized ConvNeXt-Tiny model
    with the WBCBench 13-class classification head.
    """

    model = models.convnext_tiny(
        weights=None
    )

    # Convert the existing classifier to a list.
    classifier_layers = list(
        model.classifier.children()
    )

    # Replace ImageNet 1000-class output layer
    # with our 13 WBC classes.
    classifier_layers[-1] = nn.Linear(
        CONVNEXT_FEATURES,
        NUM_CLASSES,
    )

    model.classifier = nn.Sequential(
        *classifier_layers
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# SINGLE MODEL INSTANCE
# ============================================================

# Only ONE ConvNeXt model exists in memory.
MODEL: nn.Module | None = None


# ============================================================
# LOAD FOLD 2
# ============================================================

def load_fold_2() -> nn.Module:
    """
    Load the best-performing Fold 2 checkpoint
    into the single ConvNeXt model.
    """

    global MODEL

    # --------------------------------------------------------
    # Create model only once.
    # --------------------------------------------------------

    if MODEL is None:

        print(
            "[Classifier] Creating ConvNeXt-Tiny model..."
        )

        MODEL = create_model()

        print(
            "[Classifier] Model created."
        )

    # --------------------------------------------------------
    # Verify checkpoint exists.
    # --------------------------------------------------------

    if not FOLD_2_PATH.exists():

        raise FileNotFoundError(
            "Fold 2 ConvNeXt checkpoint not found:\n"
            f"{FOLD_2_PATH}"
        )

    print(
        "[Classifier] Loading best model: Fold 2"
    )

    print(
        f"[Classifier] Checkpoint: {FOLD_2_PATH}"
    )

    # --------------------------------------------------------
    # Load checkpoint directly on CPU.
    # --------------------------------------------------------

    checkpoint = torch.load(
        FOLD_2_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # --------------------------------------------------------
    # Extract state dictionary.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported Fold 2 checkpoint format: "
            f"{type(checkpoint)}"
        )

    # --------------------------------------------------------
    # Remove DataParallel "module." prefix if present.
    # --------------------------------------------------------

    cleaned_state_dict: Dict[str, Any] = {}

    for key, value in state_dict.items():

        clean_key = key

        if clean_key.startswith("module."):

            clean_key = clean_key[
                len("module.") :
            ]

        cleaned_state_dict[
            clean_key
        ] = value

    # --------------------------------------------------------
    # Load weights.
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        MODEL.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    # --------------------------------------------------------
    # Validate checkpoint.
    # --------------------------------------------------------

    if missing_keys:

        raise RuntimeError(
            "Missing Fold 2 checkpoint keys:\n"
            + "\n".join(
                str(key)
                for key in missing_keys
            )
        )

    if unexpected_keys:

        print(
            "[Classifier] Warning: unexpected checkpoint keys:"
        )

        for key in unexpected_keys:

            print(
                f"  {key}"
            )

    # --------------------------------------------------------
    # Release checkpoint memory.
    # --------------------------------------------------------

    del cleaned_state_dict
    del state_dict
    del checkpoint

    gc.collect()

    MODEL.eval()

    print(
        "[Classifier] Fold 2 loaded successfully."
    )

    return MODEL


# ============================================================
# PUBLIC MODEL ACCESS
# ============================================================

def get_model() -> nn.Module:
    """
    Return the loaded Fold 2 model.

    This function is intentionally kept public because
    other backend code may import get_model().
    """

    return load_fold_2()


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    image: Image.Image,
) -> torch.Tensor:
    """
    Convert PIL image into a normalized ConvNeXt
    input tensor of shape [1, 3, 224, 224].
    """

    # --------------------------------------------------------
    # Ensure RGB.
    # --------------------------------------------------------

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    # --------------------------------------------------------
    # Apply preprocessing.
    # --------------------------------------------------------

    tensor = TRANSFORM(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension.
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    # --------------------------------------------------------
    # Move to CPU.
    # --------------------------------------------------------

    tensor = tensor.to(
        DEVICE
    )

    return tensor


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(
    probabilities: torch.Tensor,
) -> float:
    """
    Calculate prediction entropy.

    Lower entropy = more certain prediction.
    """

    probabilities = torch.clamp(
        probabilities,
        min=1e-8,
        max=1.0,
    )

    entropy = -torch.sum(
        probabilities
        * torch.log(probabilities)
    )

    return float(
        entropy.item()
    )


# ============================================================
# TOP-2 MARGIN
# ============================================================

def calculate_margin(
    probabilities: torch.Tensor,
) -> float:
    """
    Difference between the highest and second-highest
    class probabilities.
    """

    sorted_probabilities = torch.sort(
        probabilities,
        descending=True,
    ).values

    if sorted_probabilities.numel() < 2:

        return 0.0

    margin = (
        sorted_probabilities[0]
        - sorted_probabilities[1]
    )

    return float(
        margin.item()
    )


# ============================================================
# RELIABILITY
# ============================================================

def evaluate_reliability(
    confidence: float,
    margin: float,
    entropy: float,
) -> Dict[str, Any]:
    """
    Evaluate reliability of the Fold 2 prediction.

    Fold agreement is not required because we intentionally
    use only the best-performing Fold 2 model.
    """

    # --------------------------------------------------------
    # HIGH RELIABILITY
    # --------------------------------------------------------

    if (
        confidence >= 0.70
        and margin >= 0.20
        and entropy <= 1.20
    ):

        return {
            "status": "high",
            "reliable": True,
            "reason": (
                "Strong confidence and low "
                "prediction uncertainty."
            ),
        }

    # --------------------------------------------------------
    # MODERATE RELIABILITY
    # --------------------------------------------------------

    if (
        confidence >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
        and entropy <= ENTROPY_THRESHOLD
    ):

        return {
            "status": "moderate",
            "reliable": True,
            "reason": (
                "Acceptable confidence with "
                "moderate prediction certainty."
            ),
        }

    # --------------------------------------------------------
    # LOW RELIABILITY
    # --------------------------------------------------------

    return {
        "status": "low",
        "reliable": False,
        "reason": (
            "Prediction has lower confidence "
            "or higher uncertainty."
        ),
    }


# ============================================================
# PREDICT USING FOLD 2
# ============================================================

@torch.inference_mode()
def predict_with_fold_2(
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Run inference using ONLY Fold 2.
    """

    model = get_model()

    # --------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------

    logits = model(
        image_tensor
    )

    # --------------------------------------------------------
    # Convert logits to probabilities.
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    # --------------------------------------------------------
    # Detach from computation graph.
    # Move to CPU.
    # Clone to ensure independent memory.
    # --------------------------------------------------------

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .clone()
    )

    return probabilities


# ============================================================
# CLASSIFY ONE WBC IMAGE
# ============================================================

@torch.inference_mode()
def classify_wbc_image(
    image: Image.Image,
) -> Dict[str, Any]:
    """
    Classify one WBC image using the best-performing
    ConvNeXt Fold 2 model.
    """

    # --------------------------------------------------------
    # Prepare input.
    # --------------------------------------------------------

    tensor = prepare_image(
        image
    )

    # --------------------------------------------------------
    # Fold 2 prediction.
    # --------------------------------------------------------

    probabilities = predict_with_fold_2(
        tensor
    )

    # --------------------------------------------------------
    # Get highest probability class.
    # --------------------------------------------------------

    confidence_tensor, class_tensor = (
        torch.max(
            probabilities,
            dim=0,
        )
    )

    predicted_class = int(
        class_tensor.item()
    )

    confidence = float(
        confidence_tensor.item()
    )

    # --------------------------------------------------------
    # Class name.
    # --------------------------------------------------------

    class_name = CLASS_NAMES.get(
        predicted_class,
        "UNKNOWN",
    )

    # --------------------------------------------------------
    # Uncertainty metrics.
    # --------------------------------------------------------

    entropy = calculate_entropy(
        probabilities
    )

    margin = calculate_margin(
        probabilities
    )

    # --------------------------------------------------------
    # Since we use exactly one selected fold,
    # fold agreement is logically 1.0.
    #
    # This field is retained for frontend/API compatibility.
    # --------------------------------------------------------

    fold_agreement = 1.0

    # --------------------------------------------------------
    # Reliability.
    # --------------------------------------------------------

    reliability = evaluate_reliability(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
    )

    # --------------------------------------------------------
    # Probability table.
    # --------------------------------------------------------

    probabilities_dict: Dict[
        str,
        float,
    ] = {}

    probability_values = (
        probabilities.tolist()
    )

    for class_id, probability in enumerate(
        probability_values
    ):

        probabilities_dict[
            CLASS_NAMES[class_id]
        ] = round(
            float(probability),
            6,
        )

    # --------------------------------------------------------
    # Keep all 13 class probabilities.
    # --------------------------------------------------------

    result: Dict[str, Any] = {

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        "class_id": predicted_class,

        "class_name": class_name,

        "subtype": class_name,

        "confidence": round(
            confidence,
            6,
        ),

        # ----------------------------------------------------
        # Model decision
        # ----------------------------------------------------

        "decision_method": (
            "best_fold_2"
        ),

        "selected_fold": 2,

        "votes": 1,

        # ----------------------------------------------------
        # Fold information
        # ----------------------------------------------------

        "fold": 2,

        "fold_predictions": [
            {
                "fold": 2,

                "class_id": predicted_class,

                "class_name": class_name,

                "confidence": round(
                    confidence,
                    6,
                ),
            }
        ],

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        "margin": round(
            margin,
            6,
        ),

        "entropy": round(
            entropy,
            6,
        ),

        "fold_agreement": fold_agreement,

        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        "reliability": reliability[
            "status"
        ],

        "reliable": reliability[
            "reliable"
        ],

        "reliability_reason": reliability[
            "reason"
        ],

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        "probabilities": probabilities_dict,

        # ----------------------------------------------------
        # Model metadata
        # ----------------------------------------------------

        "model": "ConvNeXt-Tiny",

        "ensemble": "single_best_fold",

        "num_classes": NUM_CLASSES,

        "input_size": [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],

        "selected_model": "ConvNeXt Fold 2",

        "checkpoint": (
            "convnext_wbc_fold2.pth"
        ),
    }

    # --------------------------------------------------------
    # Explicit cleanup.
    # --------------------------------------------------------

    del tensor
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
    """
    Crop a WBC from the original microscopy image
    using YOLO bounding-box coordinates.
    """

    image_width, image_height = (
        image.size
    )

    # --------------------------------------------------------
    # Read bounding box.
    # --------------------------------------------------------

    x1 = float(
        bbox["x1"]
    )

    y1 = float(
        bbox["y1"]
    )

    x2 = float(
        bbox["x2"]
    )

    y2 = float(
        bbox["y2"]
    )

    # --------------------------------------------------------
    # Clamp coordinates.
    # --------------------------------------------------------

    x1 = max(
        0.0,
        min(
            x1,
            float(image_width),
        ),
    )

    y1 = max(
        0.0,
        min(
            y1,
            float(image_height),
        ),
    )

    x2 = max(
        0.0,
        min(
            x2,
            float(image_width),
        ),
    )

    y2 = max(
        0.0,
        min(
            y2,
            float(image_height),
        ),
    )

    # --------------------------------------------------------
    # Validate bounding box.
    # --------------------------------------------------------

    if x2 <= x1 or y2 <= y1:

        raise ValueError(
            "Invalid WBC bounding box."
        )

    # --------------------------------------------------------
    # Calculate padding.
    # --------------------------------------------------------

    box_width = x2 - x1

    box_height = y2 - y1

    pad_x = (
        box_width * padding
    )

    pad_y = (
        box_height * padding
    )

    # --------------------------------------------------------
    # Apply padding.
    # --------------------------------------------------------

    crop_x1 = max(
        0,
        int(
            x1 - pad_x
        ),
    )

    crop_y1 = max(
        0,
        int(
            y1 - pad_y
        ),
    )

    crop_x2 = min(
        image_width,
        int(
            x2 + pad_x
        ),
    )

    crop_y2 = min(
        image_height,
        int(
            y2 + pad_y
        ),
    )

    # --------------------------------------------------------
    # Crop image.
    # --------------------------------------------------------

    crop = image.crop(
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )
    )

    # --------------------------------------------------------
    # Ensure RGB.
    # --------------------------------------------------------

    if crop.mode != "RGB":

        crop = crop.convert(
            "RGB"
        )

    return crop


# ============================================================
# CLASSIFY WBC USING YOLO BBOX
# ============================================================

def classify_wbc_crop(
    image: Image.Image,
    bbox: Dict[str, float],
) -> Dict[str, Any]:
    """
    Crop the WBC using YOLO detection coordinates
    and classify the crop using Fold 2.
    """

    # --------------------------------------------------------
    # Crop.
    # --------------------------------------------------------

    crop = crop_wbc(
        image=image,
        bbox=bbox,
        padding=CROP_PADDING,
    )

    # --------------------------------------------------------
    # Classify.
    # --------------------------------------------------------

    result = classify_wbc_image(
        crop
    )

    # --------------------------------------------------------
    # Add crop metadata.
    # --------------------------------------------------------

    result[
        "crop_padding"
    ] = CROP_PADDING

    result[
        "crop_strategy"
    ] = "padding_15"

    result[
        "input_size"
    ] = [
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]

    return result


# ============================================================
# SIMPLE PREDICT FUNCTION
# ============================================================

def predict(
    image: Image.Image,
) -> Dict[str, Any]:
    """
    Simple public prediction function.
    """

    return classify_wbc_image(
        image
    )


# ============================================================
# STARTUP TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "CONVNEXT CLASSIFIER READY"
    )

    print(
        "=" * 60
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Model: ConvNeXt-Tiny"
    )

    print(
        "Selected model: Fold 2"
    )

    print(
        "Mode: SINGLE BEST FOLD"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print(
        f"Input size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Checkpoint: {FOLD_2_PATH}"
    )

    print()

    print(
        "Classes:"
    )

    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        print(
            f"  {class_id:2d} -> {class_name}"
        )

    print()

    print(
        "Selected Fold: 2"
    )

    print(
        "Only Fold 2 is loaded into RAM."
    )

    print(
        "3-fold ensemble disabled."
    )

    print()

    # --------------------------------------------------------
    # Verify model and checkpoint.
    # --------------------------------------------------------

    try:

        get_model()

        print(
            "Fold 2 model initialization successful."
        )

    except Exception as exc:

        print(
            "Classifier initialization failed:"
        )

        print(
            str(exc)
        )

        raise

    print(
        "=" * 60
    )