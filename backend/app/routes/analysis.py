from __future__ import annotations

import base64
import gc
import io
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

# Import modules instead of individual symbols.
# This avoids Pylance "unknown import symbol" issues and
# keeps the route compatible with the current services.
from app.services import classifier
from app.services import detector


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_IMAGES = 50

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

WBC_CLASS_NAME = "WBC"


# ============================================================
# SAFE TYPE HELPERS
# ============================================================

def safe_int(
    value: object,
    default: int = -1,
) -> int:

    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default

    return default


def safe_float(
    value: object,
    default: float = 0.0,
) -> float:

    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default

    return default


# ============================================================
# BOUNDING BOX NORMALIZATION
# ============================================================

def normalize_bbox(
    bbox: object,
) -> Optional[Dict[str, float]]:
    """
    Convert a bounding box into:

    {
        "x1": float,
        "y1": float,
        "x2": float,
        "y2": float
    }
    """

    if bbox is None:
        return None

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(bbox, dict):

        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")

        if (
            x1 is None
            or y1 is None
            or x2 is None
            or y2 is None
        ):
            return None

        return {
            "x1": safe_float(x1),
            "y1": safe_float(y1),
            "x2": safe_float(x2),
            "y2": safe_float(y2),
        }

    # --------------------------------------------------------
    # List / Tuple
    # --------------------------------------------------------

    if isinstance(bbox, (list, tuple)):

        if len(bbox) < 4:
            return None

        return {
            "x1": safe_float(bbox[0]),
            "y1": safe_float(bbox[1]),
            "x2": safe_float(bbox[2]),
            "y2": safe_float(bbox[3]),
        }

    return None


# ============================================================
# NORMALIZE ONE DETECTION
# ============================================================

def normalize_detection(
    detection: object,
) -> Optional[Dict[str, Any]]:

    if not isinstance(detection, dict):
        return None

    # --------------------------------------------------------
    # CLASS ID
    # --------------------------------------------------------

    class_id_value = detection.get("class_id")

    if class_id_value is None:
        class_id_value = detection.get("classId")

    class_id = safe_int(
        class_id_value,
        -1,
    )

    # --------------------------------------------------------
    # CLASS NAME
    # --------------------------------------------------------

    class_name_value = detection.get("class_name")

    if class_name_value is None:
        class_name_value = detection.get("className")

    if class_name_value is None:
        class_name_value = detection.get("name")

    if class_name_value is None:
        class_name = "Unknown"
    else:
        class_name = str(class_name_value)

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence_value = detection.get("confidence")

    if confidence_value is None:
        confidence_value = detection.get("conf")

    if confidence_value is None:
        confidence_value = detection.get("score")

    confidence = safe_float(
        confidence_value,
        0.0,
    )

    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    bbox_value = detection.get("bbox")

    if bbox_value is None:
        bbox_value = detection.get("box")

    if bbox_value is None:
        bbox_value = detection.get("xyxy")

    bbox = normalize_bbox(bbox_value)

    if bbox is None:
        return None

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(
            confidence,
            6,
        ),
        "bbox": {
            "x1": round(
                bbox["x1"],
                2,
            ),
            "y1": round(
                bbox["y1"],
                2,
            ),
            "x2": round(
                bbox["x2"],
                2,
            ),
            "y2": round(
                bbox["y2"],
                2,
            ),
        },
    }


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

def extract_detections(
    detector_output: object,
) -> List[Dict[str, Any]]:

    raw_detections: object = detector_output

    # --------------------------------------------------------
    # Detector may return:
    #
    # [
    #     {...},
    #     {...}
    # ]
    #
    # OR:
    #
    # {
    #     "detections": [...]
    # }
    # --------------------------------------------------------

    if isinstance(detector_output, dict):

        raw_detections = detector_output.get(
            "detections",
            [],
        )

    if not isinstance(
        raw_detections,
        list,
    ):
        return []

    detections: List[Dict[str, Any]] = []

    for raw_detection in raw_detections:

        normalized = normalize_detection(
            raw_detection
        )

        if normalized is not None:
            detections.append(
                normalized
            )

    return detections


# ============================================================
# CELL COUNTS
# ============================================================

def calculate_counts(
    detections: List[Dict[str, Any]],
) -> Dict[str, int]:

    counts = {
        "WBC": 0,
        "RBC": 0,
        "Platelets": 0,
    }

    for detection in detections:

        class_name = str(
            detection.get(
                "class_name",
                "",
            )
        ).strip().lower()

        if class_name == "wbc":

            counts["WBC"] += 1

        elif class_name == "rbc":

            counts["RBC"] += 1

        elif class_name in {
            "platelet",
            "platelets",
        }:

            counts["Platelets"] += 1

    return counts


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    image: Image.Image,
) -> str:

    rgb_image = image.convert("RGB")

    buffer = io.BytesIO()

    rgb_image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# FAILED CLASSIFICATION RESPONSE
# ============================================================

def create_failed_classification(
    reason: str,
) -> Dict[str, Any]:

    return {
        "class_id": -1,
        "class_name": "UNCERTAIN",
        "subtype": "UNCERTAIN",
        "raw_class_name": "UNCERTAIN",
        "final_decision": "UNCERTAIN",
        "confidence": 0.0,
        "margin": 0.0,
        "entropy": 0.0,
        "fold_agreement": 0.0,
        "reliability": "failed",
        "reliable": False,
        "reliability_reason": reason,
        "decision_method": "classification_failed",
        "selected_fold": 2,
        "votes": 0,
        "vote_count": 0,
        "vote_total": 1,
        "majority_vote": False,
        "fold_predictions": [],
        "probabilities": {},
        "model": "ConvNeXt-Tiny",
        "ensemble": "disabled",
        "available_folds": [2],
    }


# ============================================================
# NORMALIZE CLASSIFIER OUTPUT
# ============================================================

def normalize_classification_output(
    classification_output: object,
) -> Dict[str, Any]:

    if not isinstance(
        classification_output,
        dict,
    ):

        return create_failed_classification(
            "Classifier returned an invalid response."
        )

    result = dict(
        classification_output
    )

    # --------------------------------------------------------
    # FINAL CLASS
    # --------------------------------------------------------

    final_class_value = (
        result.get("final_decision")
        or result.get("class_name")
        or result.get("subtype")
    )

    if final_class_value is None:
        final_class_value = "UNCERTAIN"

    final_class = str(
        final_class_value
    ).strip().upper()

    if not final_class:
        final_class = "UNCERTAIN"

    result["class_name"] = final_class
    result["subtype"] = final_class
    result["final_decision"] = final_class

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    result["confidence"] = round(
        safe_float(
            result.get(
                "confidence",
                0.0,
            )
        ),
        6,
    )

    # --------------------------------------------------------
    # MARGIN
    # --------------------------------------------------------

    result["margin"] = round(
        safe_float(
            result.get(
                "margin",
                0.0,
            )
        ),
        6,
    )

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    result["entropy"] = round(
        safe_float(
            result.get(
                "entropy",
                0.0,
            )
        ),
        6,
    )

    # --------------------------------------------------------
    # SELECTED FOLD
    # --------------------------------------------------------

    result["selected_fold"] = safe_int(
        result.get(
            "selected_fold",
            2,
        ),
        2,
    )

    # --------------------------------------------------------
    # FOLD AGREEMENT
    # --------------------------------------------------------

    result["fold_agreement"] = round(
        safe_float(
            result.get(
                "fold_agreement",
                1.0,
            ),
            1.0,
        ),
        6,
    )

    # --------------------------------------------------------
    # DECISION METHOD
    # --------------------------------------------------------

    result["decision_method"] = str(
        result.get(
            "decision_method",
            "single_fold",
        )
    )

    # --------------------------------------------------------
    # VOTES
    # --------------------------------------------------------

    result["votes"] = safe_int(
        result.get(
            "votes",
            1,
        ),
        1,
    )

    result["vote_count"] = safe_int(
        result.get(
            "vote_count",
            1,
        ),
        1,
    )

    result["vote_total"] = safe_int(
        result.get(
            "vote_total",
            1,
        ),
        1,
    )

    # --------------------------------------------------------
    # FOLD PREDICTIONS
    # --------------------------------------------------------

    fold_predictions = result.get(
        "fold_predictions",
        [],
    )

    if not isinstance(
        fold_predictions,
        list,
    ):
        fold_predictions = []

    result["fold_predictions"] = fold_predictions

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = result.get(
        "probabilities",
        {},
    )

    if not isinstance(
        probabilities,
        dict,
    ):
        probabilities = {}

    result["probabilities"] = probabilities

    # --------------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------------

    result.setdefault(
        "reliability",
        "unknown",
    )

    result.setdefault(
        "reliable",
        False,
    )

    result.setdefault(
        "reliability_reason",
        "",
    )

    # --------------------------------------------------------
    # MODEL METADATA
    # --------------------------------------------------------

    result["model"] = "ConvNeXt-Tiny"

    result["ensemble"] = "disabled"

    result["available_folds"] = [2]

    result["num_classes"] = 13

    result["input_size"] = [224, 224]

    return result


# ============================================================
# WBC CLASSIFICATION SUMMARY
# ============================================================

def get_classification_summary(
    classifications: List[Dict[str, Any]],
) -> Dict[str, Any]:

    subtype_counts: Counter[str] = Counter()

    successfully_classified = 0

    classification_failures = 0

    confidence_values: List[float] = []

    # --------------------------------------------------------
    # PROCESS CLASSIFICATIONS
    # --------------------------------------------------------

    for item in classifications:

        classification_value = item.get(
            "classification",
            {},
        )

        if not isinstance(
            classification_value,
            dict,
        ):

            classification_failures += 1

            continue

        final_class_value = (
            classification_value.get(
                "final_decision"
            )
            or classification_value.get(
                "class_name"
            )
            or classification_value.get(
                "subtype"
            )
        )

        if final_class_value is None:
            final_class_value = "UNCERTAIN"

        final_class = str(
            final_class_value
        ).strip().upper()

        if not final_class:
            final_class = "UNCERTAIN"

        subtype_counts[
            final_class
        ] += 1

        confidence = safe_float(
            classification_value.get(
                "confidence",
                0.0,
            )
        )

        if confidence > 0:
            confidence_values.append(
                confidence
            )

        if final_class == "UNCERTAIN":

            classification_failures += 1

        else:

            successfully_classified += 1

    # --------------------------------------------------------
    # AVERAGE CONFIDENCE
    # --------------------------------------------------------

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        average_confidence = 0.0

    return {
        "detected_wbcs": len(
            classifications
        ),
        "successfully_classified": (
            successfully_classified
        ),
        "classification_failures": (
            classification_failures
        ),
        "subtype_counts": dict(
            subtype_counts
        ),
        "average_confidence": round(
            average_confidence,
            6,
        ),
    }


# ============================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================

@router.post(
    "/analyze",
    summary="Analyze Images",
    description=(
        "Complete blood-smear analysis pipeline. "
        "Upload one or more microscopy images. "
        "YOLOv11 detects WBC/RBC/Platelets. "
        "Detected WBC crops are classified using "
        "ConvNeXt-Tiny Fold 2 only. "
        "No 3-fold ensemble is used."
    ),
)
async def analyze_images(
    images: Annotated[
        List[UploadFile],
        File(
            ...,
            description=(
                "Upload one or more blood-smear images "
                "(JPG, JPEG, PNG, BMP, TIFF, WEBP). "
                "Maximum 50 images."
            ),
        ),
    ],
) -> Dict[str, Any]:

    """
    Complete blood-smear analysis pipeline.

    Upload
        ↓
    Image validation
        ↓
    YOLOv11 BCCD detection
        ↓
    WBC / RBC / Platelets
        ↓
    WBC bounding boxes
        ↓
    WBC crop
        ↓
    ConvNeXt-Tiny Fold 2
        ↓
    Single-fold prediction
        ↓
    JSON response
    """

    print("=" * 70)

    print(
        "[Analysis] New analysis request received."
    )

    print(
        f"[Analysis] Number of uploaded images: {len(images)}"
    )

    # ========================================================
    # VALIDATE IMAGE COUNT
    # ========================================================

    if not images:

        raise HTTPException(
            status_code=400,
            detail="No images were uploaded.",
        )

    if len(images) > MAX_IMAGES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {MAX_IMAGES} "
                "images are allowed."
            ),
        )

    # ========================================================
    # VALIDATE FILES
    # ========================================================

    invalid_files: List[str] = []

    for uploaded_image in images:

        filename = (
            uploaded_image.filename
            or ""
        )

        if not filename:

            invalid_files.append(
                "unnamed_file"
            )

            continue

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        if extension not in ALLOWED_EXTENSIONS:

            invalid_files.append(
                filename
            )

    if invalid_files:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Unsupported image format."
                ),
                "invalid_files": invalid_files,
                "allowed_formats": sorted(
                    ALLOWED_EXTENSIONS
                ),
            },
        )

    # ========================================================
    # GLOBAL COUNTS
    # ========================================================

    total_counts: Dict[str, int] = {
        "WBC": 0,
        "RBC": 0,
        "Platelets": 0,
    }

    all_wbc_classifications: List[
        Dict[str, Any]
    ] = []

    results: List[
        Dict[str, Any]
    ] = []

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_index, uploaded_image in enumerate(
        images,
        start=1,
    ):

        filename = (
            uploaded_image.filename
            or f"image_{image_index}.jpg"
        )

        print("-" * 70)

        print(
            f"[Analysis] Processing image "
            f"{image_index}/{len(images)}: {filename}"
        )

        # ====================================================
        # READ IMAGE
        # ====================================================

        try:

            image_bytes = (
                await uploaded_image.read()
            )

            if not image_bytes:

                raise ValueError(
                    "Uploaded file is empty."
                )

            if len(image_bytes) > MAX_UPLOAD_BYTES:

                raise ValueError(
                    "Image exceeds the maximum "
                    "allowed size of 15 MB."
                )

            pil_image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            ).convert("RGB")

            image_width, image_height = (
                pil_image.size
            )

            print(
                f"[Analysis] Image size: "
                f"{image_width}x{image_height}"
            )

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not read image "
                    f"'{filename}': {exc}"
                ),
            ) from exc

        # ====================================================
        # TEMPORARY FILE FOR YOLO
        # ====================================================

        temp_image_path: Optional[str] = None

        try:

            suffix = (
                Path(filename)
                .suffix
                .lower()
            )

            if not suffix:
                suffix = ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    image_bytes
                )

                temp_image_path = (
                    temp_file.name
                )

            # =================================================
            # YOLO DETECTION
            # =================================================

            print(
                "[Analysis] Running YOLO detector..."
            )

            detector_output = (
                detector.detect_image(
                    temp_image_path
                )
            )

            print(
                "[Analysis] YOLO detection completed."
            )

        except Exception as exc:

            print(
                "[Analysis] YOLO detection failed:"
            )

            print(exc)

            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        "YOLO detection failed."
                    ),
                    "filename": filename,
                    "error": str(exc),
                },
            ) from exc

        finally:

            if temp_image_path is not None:

                try:

                    Path(
                        temp_image_path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:

                    pass

        # ====================================================
        # NORMALIZE DETECTIONS
        # ====================================================

        detections = extract_detections(
            detector_output
        )

        print(
            f"[Analysis] Detections found: "
            f"{len(detections)}"
        )

        # ====================================================
        # IMAGE COUNTS
        # ====================================================

        image_counts = calculate_counts(
            detections
        )

        print(
            "[Analysis] Counts: "
            f"WBC={image_counts['WBC']}, "
            f"RBC={image_counts['RBC']}, "
            f"Platelets={image_counts['Platelets']}"
        )

        total_counts["WBC"] += (
            image_counts["WBC"]
        )

        total_counts["RBC"] += (
            image_counts["RBC"]
        )

        total_counts["Platelets"] += (
            image_counts["Platelets"]
        )

        # ====================================================
        # WBC CLASSIFICATIONS
        # ====================================================

        wbc_classifications: List[
            Dict[str, Any]
        ] = []

        for detection_index, detection in enumerate(
            detections,
            start=1,
        ):

            class_name = str(
                detection.get(
                    "class_name",
                    "",
                )
            ).strip().upper()

            # ------------------------------------------------
            # ONLY WBCs GO TO CONVNEXT
            # ------------------------------------------------

            if class_name != WBC_CLASS_NAME:

                continue

            bbox_value = detection.get(
                "bbox"
            )

            bbox = normalize_bbox(
                bbox_value
            )

            if bbox is None:

                continue

            yolo_confidence = safe_float(
                detection.get(
                    "confidence",
                    0.0,
                )
            )

            print(
                f"[Analysis] Classifying WBC "
                f"{detection_index} "
                f"with YOLO confidence "
                f"{yolo_confidence:.4f}"
            )

            # =================================================
            # CREATE WBC CROP
            # =================================================

            try:

                # Current classifier.py exposes
                # classifier.crop_wbc().
                wbc_crop = (
                    classifier.crop_wbc(
                        image=pil_image,
                        bbox=bbox,
                    )
                )

                if (
                    wbc_crop.width <= 0
                    or wbc_crop.height <= 0
                ):

                    raise ValueError(
                        "WBC crop has invalid dimensions."
                    )

            except Exception as exc:

                failed_classification = (
                    create_failed_classification(
                        f"WBC crop failed: {exc}"
                    )
                )

                wbc_result: Dict[
                    str,
                    Any
                ] = {

                    "detection_index":
                        detection_index,

                    "yolo_confidence":
                        round(
                            yolo_confidence,
                            6,
                        ),

                    "bbox":
                        bbox,

                    "classification":
                        failed_classification,

                    "crop_image":
                        None,
                }

                wbc_classifications.append(
                    wbc_result
                )

                all_wbc_classifications.append(
                    {
                        "image_index":
                            image_index,

                        "filename":
                            filename,

                        **wbc_result,
                    }
                )

                continue

            # =================================================
            # CONVNEXT FOLD 2 CLASSIFICATION
            # =================================================

            try:

                classification_output = (
                    classifier.classify_wbc_crop(
                        image=pil_image,
                        bbox=bbox,
                    )
                )

                classification_output = (
                    normalize_classification_output(
                        classification_output
                    )
                )

                print(
                    "[Analysis] Fold 2 classification: "
                    f"{classification_output.get('class_name')}"
                )

                print(
                    "[Analysis] Confidence: "
                    f"{classification_output.get('confidence')}"
                )

            except Exception as exc:

                print(
                    "[Analysis] ConvNeXt classification failed:"
                )

                print(exc)

                classification_output = (
                    create_failed_classification(
                        "ConvNeXt Fold 2 classification failed: "
                        f"{exc}"
                    )
                )

            # =================================================
            # CREATE CROP PREVIEW
            # =================================================

            crop_image_data: Optional[
                str
            ] = None

            try:

                crop_base64 = (
                    image_to_base64(
                        wbc_crop
                    )
                )

                crop_image_data = (
                    "data:image/jpeg;base64,"
                    + crop_base64
                )

            except Exception as exc:

                print(
                    "[Analysis] Could not encode WBC crop:"
                )

                print(exc)

                crop_image_data = None

            # =================================================
            # STORE RESULT
            # =================================================

            wbc_result = {

                "detection_index":
                    detection_index,

                "yolo_confidence":
                    round(
                        yolo_confidence,
                        6,
                    ),

                "bbox":
                    bbox,

                "classification":
                    classification_output,

                "crop_image":
                    crop_image_data,
            }

            wbc_classifications.append(
                wbc_result
            )

            all_wbc_classifications.append(
                {
                    "image_index":
                        image_index,

                    "filename":
                        filename,

                    **wbc_result,
                }
            )

        # ====================================================
        # IMAGE WBC SUMMARY
        # ====================================================

        image_wbc_summary = (
            get_classification_summary(
                wbc_classifications
            )
        )

        # ====================================================
        # IMAGE RESULT
        # ====================================================

        image_result: Dict[
            str,
            Any
        ] = {

            "index":
                image_index,

            "filename":
                filename,

            "content_type":
                (
                    uploaded_image.content_type
                    or "application/octet-stream"
                ),

            "image_size": {

                "width":
                    image_width,

                "height":
                    image_height,
            },

            "counts":
                image_counts,

            "detection_count":
                len(detections),

            "detections":
                detections,

            "wbc_classifications":
                wbc_classifications,

            "wbc_subtype_analysis":
                image_wbc_summary,
        }

        results.append(
            image_result
        )

        # ====================================================
        # RELEASE IMAGE MEMORY
        # ====================================================

        del image_bytes

        del pil_image

        gc.collect()

    # ========================================================
    # GLOBAL WBC SUMMARY
    # ========================================================

    global_wbc_summary = (
        get_classification_summary(
            all_wbc_classifications
        )
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    response: Dict[
        str,
        Any
    ] = {

        "status":
            "success",

        "message": (
            "Images analyzed successfully "
            "using YOLOv11 BCCD detection and "
            "ConvNeXt-Tiny Fold 2 single-fold "
            "WBC classification."
        ),

        "image_count":
            len(images),

        "max_images":
            MAX_IMAGES,

        # ----------------------------------------------------
        # CELL COUNTS
        # ----------------------------------------------------

        "total_counts":
            total_counts,

        # ----------------------------------------------------
        # GLOBAL WBC SUBTYPE ANALYSIS
        # ----------------------------------------------------

        "wbc_subtype_analysis":
            global_wbc_summary,

        # ----------------------------------------------------
        # IMAGE RESULTS
        # ----------------------------------------------------

        "results":
            results,

        # ----------------------------------------------------
        # INFERENCE INFORMATION
        # ----------------------------------------------------

        "inference": {

            "detector": {

                "model":
                    "yolo11s_bccd_best.pt",

                "task":
                    "BCCD blood-cell detection",

                "confidence_threshold":
                    0.25,

                "iou_threshold":
                    0.45,
            },

            # =================================================
            # CLASSIFIER
            # =================================================

            "classifier": {

                "model":
                    "ConvNeXt-Tiny",

                "ensemble":
                    "disabled",

                "folds":
                    1,

                "selected_fold":
                    2,

                "available_folds":
                    [2],

                "classes":
                    13,

                "input_size":
                    "224x224",

                "feature_dimension":
                    768,

                "task":
                    "WBC subtype classification",

                "decision_method":
                    "single_fold",

                "decision_rule":
                    (
                        "Prediction from selected "
                        "best-performing Fold 2."
                    ),

                "majority_voting":
                    False,

                "status":
                    "completed",
            },

            # =================================================
            # PIPELINE
            # =================================================

            "pipeline": {

                "stage_1":
                    "YOLOv11 BCCD detection",

                "stage_2":
                    "WBC/RBC/Platelet detection",

                "stage_3":
                    "WBC bounding-box extraction",

                "stage_4":
                    "WBC crop generation",

                "stage_5":
                    "ConvNeXt-Tiny Fold 2 inference",

                "stage_6":
                    "Single-fold prediction",

                "stage_7":
                    "JSON response",
            },
        },
    }

    print("=" * 70)

    print(
        "[Analysis] Analysis completed successfully."
    )

    print(
        f"[Analysis] Images: {len(images)}"
    )

    print(
        f"[Analysis] WBC: {total_counts['WBC']}"
    )

    print(
        f"[Analysis] RBC: {total_counts['RBC']}"
    )

    print(
        f"[Analysis] Platelets: {total_counts['Platelets']}"
    )

    print(
        f"[Analysis] Classified WBCs: "
        f"{global_wbc_summary['successfully_classified']}"
    )

    print(
        "[Analysis] Classifier: "
        "ConvNeXt-Tiny Fold 2"
    )

    print(
        "[Analysis] Ensemble: DISABLED"
    )

    print("=" * 70)

    return response