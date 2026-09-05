# ============================================================
# BloodCell Intelligence
# WBCBench ConvNeXt-Tiny Classifier
#
# RENDER 512 MB MEMORY-OPTIMIZED VERSION
#
# IMPORTANT:
# ONLY FOLD 2 IS USED
# NO 3-FOLD ENSEMBLE
# ============================================================

from pathlib import Path
from typing import Any, Dict, List

import gc

import torch
import torch.nn as nn

from PIL import Image

from torchvision import models, transforms


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# PYTORCH CPU MEMORY OPTIMIZATION
# ============================================================

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


# ============================================================
# IMPORTANT MEMORY OPTIMIZATION
#
# ONLY FOLD 2
#
# The old classifier loaded/processed:
#
#     fold1
#     fold2
#     fold3
#
# We now use ONLY:
#
#     fold2
# ============================================================

SELECTED_FOLD = 2

CHECKPOINT_PATH = (
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
# CREATE CONVNEXT-TINY MODEL
# ============================================================

def create_model() -> nn.Module:

    print(
        "[Classifier] Creating ConvNeXt-Tiny..."
    )

    model = models.convnext_tiny(
        weights=None
    )

    # --------------------------------------------------------
    # ConvNeXt-Tiny classifier:
    #
    # Layer 0 -> LayerNorm
    # Layer 1 -> Flatten
    # Layer 2 -> Linear
    #
    # Replace final Linear:
    #
    # 768 -> 13
    # --------------------------------------------------------

    classifier_layers = list(
        model.classifier.children()
    )

    classifier_layers[-1] = nn.Linear(
        CONVNEXT_FEATURES,
        NUM_CLASSES,
    )

    model.classifier = nn.Sequential(
        *classifier_layers
    )

    # CPU only
    model = model.to(DEVICE)

    # Evaluation mode
    model.eval()

    print(
        "[Classifier] ConvNeXt-Tiny created."
    )

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

    print(
        "[Classifier] Loading Fold 2 checkpoint..."
    )

    print(
        f"[Classifier] Path: {checkpoint_path}"
    )

    # --------------------------------------------------------
    # Load checkpoint directly on CPU.
    #
    # weights_only=True avoids loading unnecessary
    # training objects.
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:

        # Compatibility with older PyTorch
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

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
            "Unsupported ConvNeXt checkpoint format: "
            f"{type(checkpoint)}"
        )

    # --------------------------------------------------------
    # Remove DataParallel prefix if present.
    #
    # Only create a new dictionary if necessary.
    # --------------------------------------------------------

    has_module_prefix = any(
        key.startswith("module.")
        for key in state_dict.keys()
    )

    if has_module_prefix:

        cleaned_state_dict = {
            (
                key[len("module."):]
                if key.startswith("module.")
                else key
            ): value
            for key, value in state_dict.items()
        }

    else:

        cleaned_state_dict = state_dict

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if missing_keys:

        raise RuntimeError(
            "Missing ConvNeXt Fold 2 checkpoint keys:\n"
            + "\n".join(
                str(key)
                for key in missing_keys
            )
        )

    if unexpected_keys:

        print(
            "[Classifier] Warning: unexpected "
            "checkpoint keys:"
        )

        for key in unexpected_keys:

            print(
                f"  {key}"
            )

    # --------------------------------------------------------
    # Release temporary checkpoint memory
    # --------------------------------------------------------

    if cleaned_state_dict is not state_dict:

        del cleaned_state_dict

    del state_dict
    del checkpoint

    gc.collect()

    model.eval()

    print(
        "[Classifier] Fold 2 loaded successfully."
    )

    return model


# ============================================================
# SINGLE MODEL INSTANCE
# ============================================================
#
# IMPORTANT:
#
# Only ONE ConvNeXt model exists.
#
# Fold 1 -> NOT loaded
# Fold 2 -> loaded
# Fold 3 -> NOT loaded
#
# ============================================================

MODEL: nn.Module | None = None


def get_model() -> nn.Module:

    global MODEL

    if MODEL is None:

        print(
            "[Classifier] Initializing "
            "ONLY Fold 2..."
        )

        # ----------------------------------------------------
        # Create model architecture
        # ----------------------------------------------------

        MODEL = create_model()

        # ----------------------------------------------------
        # Load Fold 2 weights
        # ----------------------------------------------------

        MODEL = load_checkpoint(
            MODEL,
            CHECKPOINT_PATH,
        )

        print(
            "[Classifier] Single Fold 2 model "
            "is ready."
        )

    return MODEL


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    image: Image.Image,
) -> torch.Tensor:

    # --------------------------------------------------------
    # Ensure RGB
    # --------------------------------------------------------

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = TRANSFORM(
        image
    )

    # --------------------------------------------------------
    # Ensure float32
    # --------------------------------------------------------

    tensor = torch.as_tensor(
        tensor,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    # --------------------------------------------------------
    # CPU
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
# FOLD AGREEMENT
# ============================================================
#
# Since only one fold is used:
#
# Fold agreement = 1.0
#
# There is no disagreement because there is
# only one model prediction.
# ============================================================

def calculate_fold_agreement(
    predictions: List[int],
) -> float:

    if not predictions:

        return 0.0

    # One selected fold = complete agreement
    if len(predictions) == 1:

        return 1.0

    counts: Dict[int, int] = {}

    for prediction in predictions:

        counts[prediction] = (
            counts.get(
                prediction,
                0,
            )
            + 1
        )

    maximum_count = max(
        counts.values()
    )

    agreement = (
        maximum_count
        / len(predictions)
    )

    return float(
        agreement
    )


# ============================================================
# RELIABILITY
# ============================================================

def evaluate_reliability(
    confidence: float,
    margin: float,
    entropy: float,
    fold_agreement: float,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # HIGH
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
                "prediction uncertainty from "
                "the selected Fold 2 model."
            ),
        }

    # --------------------------------------------------------
    # MODERATE
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
                "Acceptable confidence and "
                "prediction separation from "
                "the selected Fold 2 model."
            ),
        }

    # --------------------------------------------------------
    # LOW
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
# CLASSIFY ONE WBC IMAGE
# ============================================================

@torch.inference_mode()
def classify_wbc_image(
    image: Image.Image,
) -> Dict[str, Any]:

    print(
        "[Classifier] Starting Fold 2 prediction..."
    )

    # --------------------------------------------------------
    # Get ONLY Fold 2 model
    # --------------------------------------------------------

    model = get_model()

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    tensor = prepare_image(
        image
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    # --------------------------------------------------------
    # Move only the tiny 13-value result to CPU
    # --------------------------------------------------------

    probabilities = (
        probabilities
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # Get predicted class
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

    class_name = CLASS_NAMES.get(
        predicted_class,
        "UNKNOWN",
    )

    # --------------------------------------------------------
    # Uncertainty metrics
    # --------------------------------------------------------

    entropy = calculate_entropy(
        probabilities
    )

    margin = calculate_margin(
        probabilities
    )

    # --------------------------------------------------------
    # Single-fold information
    # --------------------------------------------------------

    fold_predictions = [
        predicted_class
    ]

    fold_confidences = [
        confidence
    ]

    fold_agreement = (
        calculate_fold_agreement(
            fold_predictions
        )
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    reliability = evaluate_reliability(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        fold_agreement=fold_agreement,
    )

    # --------------------------------------------------------
    # Probability table
    # --------------------------------------------------------

    probability_values = (
        probabilities.tolist()
    )

    probabilities_dict: Dict[
        str,
        float,
    ] = {}

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
    # Single-fold result
    #
    # Keep "fold_predictions" structure so the
    # existing frontend does not need major changes.
    # --------------------------------------------------------

    fold_results = [

        {
            "fold":
                SELECTED_FOLD,

            "class_id":
                predicted_class,

            "class_name":
                class_name,

            "confidence":
                round(
                    confidence,
                    6,
                ),
        }

    ]

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result: Dict[str, Any] = {

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        "class_id":
            predicted_class,

        "class_name":
            class_name,

        "subtype":
            class_name,

        "confidence":
            round(
                confidence,
                6,
            ),

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        "decision_method":
            "single_fold",

        "selected_fold":
            SELECTED_FOLD,

        "votes":
            1,

        # ----------------------------------------------------
        # Compatibility fields
        # ----------------------------------------------------
        #
        # These are retained so the frontend can continue
        # reading the response without breaking.
        # ----------------------------------------------------

        "ensemble_class_id":
            predicted_class,

        "ensemble_class_name":
            class_name,

        "ensemble_confidence":
            round(
                confidence,
                6,
            ),

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        "margin":
            round(
                margin,
                6,
            ),

        "entropy":
            round(
                entropy,
                6,
            ),

        "fold_agreement":
            round(
                fold_agreement,
                6,
            ),

        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        "reliability":
            reliability["status"],

        "reliable":
            reliability["reliable"],

        "reliability_reason":
            reliability["reason"],

        # ----------------------------------------------------
        # Fold results
        # ----------------------------------------------------

        "fold_predictions":
            fold_results,

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        "probabilities":
            probabilities_dict,

        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        "model":
            "ConvNeXt-Tiny",

        "ensemble":
            "disabled",

        "num_classes":
            NUM_CLASSES,

        "input_size":
            IMAGE_SIZE,

        "available_folds":
            [SELECTED_FOLD],

        "fold":
            SELECTED_FOLD,
    }

    # --------------------------------------------------------
    # Explicit memory cleanup
    # --------------------------------------------------------

    del tensor
    del logits
    del probabilities
    del confidence_tensor
    del class_tensor

    gc.collect()

    print(
        "[Classifier] Fold 2 prediction complete."
    )

    return result


# ============================================================
# CROP WBC FROM YOLO BOUNDING BOX
# ============================================================

def crop_wbc(
    image: Image.Image,
    bbox: Dict[str, float],
    padding: float = CROP_PADDING,
) -> Image.Image:

    image_width, image_height = (
        image.size
    )

    # --------------------------------------------------------
    # Coordinates
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
    # Clamp coordinates
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
    # Validate bounding box
    # --------------------------------------------------------

    if x2 <= x1 or y2 <= y1:

        raise ValueError(
            "Invalid WBC bounding box."
        )

    # --------------------------------------------------------
    # Calculate padding
    # --------------------------------------------------------

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    pad_x = (
        box_width * padding
    )

    pad_y = (
        box_height * padding
    )

    # --------------------------------------------------------
    # Apply padding
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
    # Crop
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
    # Ensure RGB
    # --------------------------------------------------------

    if crop.mode != "RGB":

        crop = crop.convert(
            "RGB"
        )

    return crop


# ============================================================
# CLASSIFY WBC CROP
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

    try:

        result = classify_wbc_image(
            crop
        )

    finally:

        # ----------------------------------------------------
        # Release PIL crop
        # ----------------------------------------------------

        crop.close()

        del crop

        gc.collect()

    # --------------------------------------------------------
    # Add crop metadata
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

    return classify_wbc_image(
        image
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict[str, Any]:

    # --------------------------------------------------------
    # Loading the model here is intentional.
    # --------------------------------------------------------

    model = get_model()

    return {

        "model":
            "ConvNeXt-Tiny",

        "selected_fold":
            SELECTED_FOLD,

        "available_folds":
            [SELECTED_FOLD],

        "ensemble":
            "disabled",

        "folds":
            1,

        "classes":
            CLASS_NAMES,

        "num_classes":
            NUM_CLASSES,

        "input_size":
            IMAGE_SIZE,

        "feature_dimension":
            CONVNEXT_FEATURES,

        "device":
            str(DEVICE),

        "task":
            "WBC subtype classification",

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "margin_threshold":
            MARGIN_THRESHOLD,

        "entropy_threshold":
            ENTROPY_THRESHOLD,

        "checkpoint":
            CHECKPOINT_PATH.name,

        "status":
            "ready",
    }


# ============================================================
# STARTUP TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "CONVNEXT WBC CLASSIFIER"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Memory mode: SINGLE FOLD"
    )

    print(
        f"Selected fold: {SELECTED_FOLD}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print(
        f"Input size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        "Ensemble: DISABLED"
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
        "Loading Fold 2..."
    )

    try:

        get_model()

        print()

        print(
            "Fold 2 model loaded successfully."
        )

    except Exception as error:

        print()

        print(
            "ERROR:"
        )

        print(
            error
        )

    print("=" * 60)