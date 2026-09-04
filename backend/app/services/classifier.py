# ============================================================
# BloodCell Intelligence
# WBCBench ConvNeXt 3-Fold Ensemble Classifier
# ============================================================

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("WBC CONVNEXT CLASSIFIER")
print("=" * 60)
print(f"Device: {DEVICE}")


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

# ConvNeXt-Tiny feature dimension
CONVNEXT_FEATURES = 768

IMAGE_SIZE = 224

CROP_PADDING = 0.15


# ============================================================
# RELIABILITY THRESHOLDS
#
# IMPORTANT:
# These thresholds are ONLY used for metadata.
# They DO NOT convert the final prediction to UNCERTAIN.
# ============================================================

CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.15
ENTROPY_THRESHOLD = 1.60


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = BASE_DIR / "weights"


FOLD_PATHS: List[Path] = [
    WEIGHTS_DIR / "convnext_wbc_fold1.pth",
    WEIGHTS_DIR / "convnext_wbc_fold2.pth",
    WEIGHTS_DIR / "convnext_wbc_fold3.pth",
]


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

    model = models.convnext_tiny(
        weights=None
    )

    # ConvNeXt-Tiny classifier:
    #
    # Layer 0 -> LayerNorm
    # Layer 1 -> Flatten
    # Layer 2 -> Linear
    #
    # We replace the final Linear layer with:
    #
    # 768 -> 13
    #

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

    model = model.to(
        DEVICE
    )

    model.eval()

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
            "ConvNeXt checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
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
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict: Dict[str, Any] = {}

    for key, value in state_dict.items():

        clean_key = key

        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[
                len("module.") :
            ]

        cleaned_state_dict[
            clean_key
        ] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing_keys:

        raise RuntimeError(
            "Missing ConvNeXt checkpoint keys:\n"
            + "\n".join(
                str(key)
                for key in missing_keys
            )
        )

    if unexpected_keys:

        print(
            "Warning: unexpected checkpoint keys:"
        )

        for key in unexpected_keys:

            print(
                f"  {key}"
            )

    model.eval()

    return model


# ============================================================
# LOAD ALL THREE FOLDS
# ============================================================

MODELS: List[nn.Module] = []


print()
print("=" * 60)
print("LOADING CONVNEXT FOLDS")
print("=" * 60)


for fold_index, checkpoint_path in enumerate(
    FOLD_PATHS,
    start=1,
):

    print()

    print(
        f"Loading Fold {fold_index}..."
    )

    model = create_model()

    model = load_checkpoint(
        model,
        checkpoint_path,
    )

    MODELS.append(
        model
    )

    print(
        f"Fold {fold_index} loaded successfully."
    )


print()
print(
    f"Total folds loaded: {len(MODELS)}"
)

print(
    f"Number of classes: {NUM_CLASSES}"
)

print(
    f"Feature dimension: {CONVNEXT_FEATURES}"
)

print(
    f"Input size: {IMAGE_SIZE}x{IMAGE_SIZE}"
)

print("=" * 60)


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

def calculate_fold_agreement(
    predictions: List[int],
) -> float:

    if not predictions:

        return 0.0

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
#
# IMPORTANT:
# This function DOES NOT determine the final class.
# It only describes confidence/reliability.
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
        and fold_agreement >= 0.66
    ):

        return {
            "status": "high",
            "reliable": True,
            "reason": (
                "Strong confidence, low uncertainty "
                "and strong fold agreement."
            ),
        }

    # --------------------------------------------------------
    # MODERATE
    # --------------------------------------------------------

    if (
        confidence >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
        and entropy <= ENTROPY_THRESHOLD
        and fold_agreement >= 0.66
    ):

        return {
            "status": "moderate",
            "reliable": True,
            "reason": (
                "Acceptable confidence and "
                "fold agreement."
            ),
        }

    # --------------------------------------------------------
    # LOW
    #
    # Still NOT UNCERTAIN.
    # The class prediction remains valid.
    # --------------------------------------------------------

    return {
        "status": "low",
        "reliable": False,
        "reason": (
            "Prediction has lower reliability metrics, "
            "but the fold decision rule still determines "
            "the final predicted subtype."
        ),
    }


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
    # Explicit Tensor conversion
    # --------------------------------------------------------

    tensor = torch.as_tensor(
        tensor
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    tensor = tensor.to(
        DEVICE
    )

    return tensor


# ============================================================
# FINAL 3-FOLD DECISION
#
# RULE:
#
# 3 same:
#     majority = same class
#
# 2 same:
#     majority = repeated class
#
# 3 different:
#     choose class from fold with highest confidence
# ============================================================

def determine_final_prediction(
    fold_predictions: List[int],
    fold_confidences: List[float],
) -> Dict[str, Any]:

    if len(fold_predictions) == 0:

        raise RuntimeError(
            "No fold predictions available."
        )

    if len(fold_predictions) != len(
        fold_confidences
    ):

        raise RuntimeError(
            "Fold predictions and confidence "
            "lists have different lengths."
        )

    # --------------------------------------------------------
    # Count votes
    # --------------------------------------------------------

    vote_counts: Dict[int, int] = {}

    for class_id in fold_predictions:

        vote_counts[class_id] = (
            vote_counts.get(
                class_id,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # Find highest vote count
    # --------------------------------------------------------

    max_votes = max(
        vote_counts.values()
    )

    majority_classes = [
        class_id
        for class_id, count
        in vote_counts.items()
        if count == max_votes
    ]

    # ========================================================
    # CASE 1:
    # Majority exists
    #
    # Example:
    #
    # PC
    # PC
    # LY
    #
    # -> PC
    #
    # ========================================================

    if max_votes >= 2:

        final_class = majority_classes[0]

        # Average confidence of folds
        # that voted for the final class.

        winning_confidences = [
            fold_confidences[index]
            for index, class_id
            in enumerate(fold_predictions)
            if class_id == final_class
        ]

        if winning_confidences:

            final_confidence = (
                sum(winning_confidences)
                / len(winning_confidences)
            )

        else:

            final_confidence = 0.0

        return {
            "class_id": final_class,
            "confidence": float(
                final_confidence
            ),
            "decision_method": (
                "majority_vote"
            ),
            "votes": max_votes,
        }

    # ========================================================
    # CASE 2:
    # ALL THREE DIFFERENT
    #
    # Example:
    #
    # Fold 1 -> SNE -> 0.288949
    # Fold 2 -> BNE -> 0.268687
    # Fold 3 -> MO  -> 0.236619
    #
    # Highest = SNE
    #
    # -> SNE
    #
    # ========================================================

    highest_confidence_index = max(
        range(
            len(fold_confidences)
        ),
        key=lambda index: fold_confidences[
            index
        ],
    )

    final_class = fold_predictions[
        highest_confidence_index
    ]

    final_confidence = fold_confidences[
        highest_confidence_index
    ]

    return {
        "class_id": final_class,
        "confidence": float(
            final_confidence
        ),
        "decision_method": (
            "highest_fold_probability"
        ),
        "selected_fold": (
            highest_confidence_index + 1
        ),
        "votes": 1,
    }


# ============================================================
# CLASSIFY ONE WBC IMAGE
# ============================================================

@torch.inference_mode()
def classify_wbc_image(
    image: Image.Image,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    tensor = prepare_image(
        image
    )

    fold_predictions: List[int] = []

    fold_confidences: List[float] = []

    fold_probabilities: List[
        torch.Tensor
    ] = []

    # ========================================================
    # RUN ALL THREE FOLDS
    # ========================================================

    for fold_index, model in enumerate(
        MODELS,
        start=1,
    ):

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

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

        fold_predictions.append(
            predicted_class
        )

        fold_confidences.append(
            confidence
        )

        fold_probabilities.append(
            probabilities
        )

    # ========================================================
    # ENSEMBLE PROBABILITIES
    #
    # Average probabilities from all folds.
    #
    # This is still returned for analysis/frontend.
    # ========================================================

    stacked_probabilities = torch.stack(
        fold_probabilities,
        dim=0,
    )

    ensemble_probabilities = (
        stacked_probabilities.mean(
            dim=0
        )
    )

    # ========================================================
    # OLD ENSEMBLE ARGMAX
    #
    # Kept as diagnostic information.
    # ========================================================

    ensemble_confidence_tensor, ensemble_class_tensor = (
        torch.max(
            ensemble_probabilities,
            dim=0,
        )
    )

    ensemble_class = int(
        ensemble_class_tensor.item()
    )

    ensemble_confidence = float(
        ensemble_confidence_tensor.item()
    )

    # ========================================================
    # NEW FINAL DECISION
    # ========================================================

    final_decision = determine_final_prediction(
        fold_predictions=fold_predictions,
        fold_confidences=fold_confidences,
    )

    predicted_class = int(
        final_decision[
            "class_id"
        ]
    )

    final_confidence = float(
        final_decision[
            "confidence"
        ]
    )

    decision_method = str(
        final_decision[
            "decision_method"
        ]
    )

    # ========================================================
    # FINAL CLASS NAME
    # ========================================================

    final_class_name = CLASS_NAMES.get(
        predicted_class,
        "UNKNOWN",
    )

    # ========================================================
    # RAW ENSEMBLE CLASS
    # ========================================================

    ensemble_class_name = CLASS_NAMES.get(
        ensemble_class,
        "UNKNOWN",
    )

    # ========================================================
    # UNCERTAINTY METRICS
    # ========================================================

    entropy = calculate_entropy(
        ensemble_probabilities
    )

    margin = calculate_margin(
        ensemble_probabilities
    )

    fold_agreement = (
        calculate_fold_agreement(
            fold_predictions
        )
    )

    # ========================================================
    # RELIABILITY
    #
    # Metadata only.
    # It does NOT change final_class_name.
    # ========================================================

    reliability = evaluate_reliability(
        confidence=final_confidence,
        margin=margin,
        entropy=entropy,
        fold_agreement=fold_agreement,
    )

    # ========================================================
    # PROBABILITY TABLE
    # ========================================================

    probabilities_dict: Dict[
        str,
        float
    ] = {}

    probability_values = (
        ensemble_probabilities.tolist()
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

    # ========================================================
    # FOLD RESULTS
    # ========================================================

    fold_results: List[
        Dict[str, Any]
    ] = []

    for index in range(
        len(MODELS)
    ):

        class_id = fold_predictions[
            index
        ]

        fold_results.append(
            {
                "fold": index + 1,

                "class_id": class_id,

                "class_name": CLASS_NAMES.get(
                    class_id,
                    "UNKNOWN",
                ),

                "confidence": round(
                    fold_confidences[
                        index
                    ],
                    6,
                ),
            }
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result: Dict[str, Any] = {

        # ----------------------------------------------------
        # NEW FINAL PREDICTION
        # ----------------------------------------------------

        "class_id": predicted_class,

        "class_name": final_class_name,

        "subtype": final_class_name,

        "confidence": round(
            final_confidence,
            6,
        ),

        # ----------------------------------------------------
        # Decision information
        # ----------------------------------------------------

        "decision_method": decision_method,

        "votes": final_decision.get(
            "votes",
            1,
        ),

        # ----------------------------------------------------
        # Diagnostic ensemble information
        # ----------------------------------------------------

        "ensemble_class_id": ensemble_class,

        "ensemble_class_name": (
            ensemble_class_name
        ),

        "ensemble_confidence": round(
            ensemble_confidence,
            6,
        ),

        # ----------------------------------------------------
        # Uncertainty metrics
        # ----------------------------------------------------

        "margin": round(
            margin,
            6,
        ),

        "entropy": round(
            entropy,
            6,
        ),

        "fold_agreement": round(
            fold_agreement,
            6,
        ),

        # ----------------------------------------------------
        # Reliability metadata
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
        # Individual folds
        # ----------------------------------------------------

        "fold_predictions": fold_results,

        # ----------------------------------------------------
        # Average probability distribution
        # ----------------------------------------------------

        "probabilities": probabilities_dict,

        # ----------------------------------------------------
        # Model metadata
        # ----------------------------------------------------

        "model": "ConvNeXt-Tiny",

        "ensemble": "3-fold",

        "num_classes": NUM_CLASSES,

        "input_size": IMAGE_SIZE,
    }

    # --------------------------------------------------------
    # If all folds differ, include selected fold information.
    # --------------------------------------------------------

    if decision_method == (
        "highest_fold_probability"
    ):

        result[
            "selected_fold"
        ] = final_decision.get(
            "selected_fold"
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

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    image_width, image_height = (
        image.size
    )

    # --------------------------------------------------------
    # Bounding box
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
    # Validate bbox
    # --------------------------------------------------------

    if x2 <= x1 or y2 <= y1:

        raise ValueError(
            "Invalid WBC bounding box."
        )

    # --------------------------------------------------------
    # Calculate padding
    # --------------------------------------------------------

    box_width = x2 - x1

    box_height = y2 - y1

    pad_x = (
        box_width
        * padding
    )

    pad_y = (
        box_height
        * padding
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
# CLASSIFY WBC USING YOLO BBOX
# ============================================================

def classify_wbc_crop(
    image: Image.Image,
    bbox: Dict[str, float],
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Extract WBC
    # --------------------------------------------------------

    crop = crop_wbc(
        image=image,
        bbox=bbox,
        padding=CROP_PADDING,
    )

    # --------------------------------------------------------
    # Classify
    # --------------------------------------------------------

    result = classify_wbc_image(
        crop
    )

    # --------------------------------------------------------
    # Crop information
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
# STARTUP TEST
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        "CONVNEXT CLASSIFIER READY"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Folds loaded: {len(MODELS)}"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print(
        f"Feature dimension: {CONVNEXT_FEATURES}"
    )

    print(
        f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print()

    print(
        "Classes:"
    )

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id:2d} -> {class_name}"
        )

    print()

    print(
        "Decision rule:"
    )

    print(
        "  3 same      -> majority result"
    )

    print(
        "  2 same      -> majority result"
    )

    print(
        "  3 different -> highest fold probability"
    )

    print()

    print(
        "Classifier initialization successful."
    )

    print("=" * 60)