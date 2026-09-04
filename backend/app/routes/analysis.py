from __future__ import annotations

import base64
import gc
import io
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.services.detector import detect_image
from app.services.classifier import (
    classify_wbc_crop,
    crop_wbc,
)


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

# Maximum number of images accepted by the API.
MAX_IMAGES = 50

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

# Keep crop previews small.
# This reduces the JSON response size and memory usage.
MAX_CROP_PREVIEW_SIZE = 384

# JPEG quality for returned WBC crop previews.
CROP_JPEG_QUALITY = 70


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
# NORMALIZE BOUNDING BOX
# ============================================================

def normalize_bbox(
    bbox: object,
) -> Optional[Dict[str, float]]:

    if bbox is None:
        return None

    # Dictionary format
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

    # List / tuple format
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
    # Class ID
    # --------------------------------------------------------

    class_id_value = detection.get("class_id")

    if class_id_value is None:
        class_id_value = detection.get("classId")

    class_id = safe_int(
        class_id_value,
        -1,
    )

    # --------------------------------------------------------
    # Class name
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
    # Confidence
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
    # Bounding box
    # --------------------------------------------------------

    bbox_value = detection.get("bbox")

    if bbox_value is None:
        bbox_value = detection.get("box")

    if bbox_value is None:
        bbox_value = detection.get("xyxy")

    bbox = normalize_bbox(
        bbox_value
    )

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

    if isinstance(
        detector_output,
        dict,
    ):

        raw_detections = detector_output.get(
            "detections",
            [],
        )

    if not isinstance(
        raw_detections,
        list,
    ):
        return []

    detections: List[
        Dict[str, Any]
    ] = []

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
# CALCULATE CELL COUNTS
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
# CREATE SMALL BASE64 CROP
# ============================================================

def image_to_base64(
    image: Image.Image,
) -> str:

    # Work on a copy so we do not accidentally
    # modify the original crop.
    preview = image.convert("RGB")

    # Resize large WBC previews.
    preview.thumbnail(
        (
            MAX_CROP_PREVIEW_SIZE,
            MAX_CROP_PREVIEW_SIZE,
        ),
        Image.Resampling.LANCZOS,
    )

    buffer = io.BytesIO()

    preview.save(
        buffer,
        format="JPEG",
        quality=CROP_JPEG_QUALITY,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    preview.close()
    buffer.close()

    return encoded


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
        "ensemble_confidence": 0.0,

        "margin": 0.0,
        "entropy": 0.0,

        # One Fold is used.
        "fold_agreement": 1.0,

        "reliability": "failed",
        "reliable": False,
        "reliability_reason": reason,

        "decision_method": "classification_failed",

        "votes": 1,
        "vote_count": 1,
        "vote_total": 1,
        "majority_vote": True,

        "fold_predictions": [],
        "probabilities": {},
    }


# ============================================================
# NORMALIZE CLASSIFIER OUTPUT
# ============================================================

def normalize_classification_output(
    classification: object,
) -> Dict[str, Any]:

    if not isinstance(
        classification,
        dict,
    ):

        return create_failed_classification(
            "Classifier returned an invalid response."
        )

    result = dict(
        classification
    )

    # --------------------------------------------------------
    # Final class
    # --------------------------------------------------------

    final_class = (
        result.get("final_decision")
        or result.get("class_name")
        or result.get("subtype")
    )

    if final_class is None:
        final_class = "UNCERTAIN"

    final_class = str(
        final_class
    ).strip().upper()

    if not final_class:
        final_class = "UNCERTAIN"

    result["class_name"] = final_class
    result["subtype"] = final_class
    result["final_decision"] = final_class

    # --------------------------------------------------------
    # Confidence
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
    # One-fold agreement
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

    # One fold = one prediction.
    result["vote_total"] = 1

    if result.get("votes") is None:
        result["votes"] = 1

    if result.get("vote_count") is None:
        result["vote_count"] = 1

    # --------------------------------------------------------
    # Fold predictions
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
    # Probabilities
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
    # Decision method
    # --------------------------------------------------------

    if not result.get(
        "decision_method"
    ):

        result[
            "decision_method"
        ] = "single_fold_prediction"

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    if "reliability" not in result:
        result["reliability"] = "unknown"

    if "reliable" not in result:
        result["reliable"] = False

    if "reliability_reason" not in result:
        result["reliability_reason"] = ""

    return result


# ============================================================
# CLASSIFICATION SUMMARY
# ============================================================

def get_classification_summary(
    classifications: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    subtype_counts: Counter[str] = Counter()

    successfully_classified = 0
    classification_failures = 0

    confidence_values: List[
        float
    ] = []

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
            or "UNCERTAIN"
        )

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

@router.post("/analyze")
async def analyze_images(
    images: List[UploadFile] = File(...),
) -> Dict[str, Any]:

    """
    Memory-optimized blood-smear analysis.

    Pipeline:

        Upload
          ↓
        YOLOv11
          ↓
        WBC / RBC / Platelets
          ↓
        WBC crop
          ↓
        ConvNeXt Fold 2 ONLY
          ↓
        JSON response

    Important:
        Only one classifier fold is used.
        Images are processed sequentially.
        No MODEL_READY or release_model imports are used.
    """

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
    # VALIDATE EXTENSIONS
    # ========================================================

    invalid_files: List[str] = []

    for uploaded_image in images:

        filename = (
            uploaded_image.filename
            or ""
        )

        if "." not in filename:

            invalid_files.append(
                filename
            )

            continue

        extension = (
            "."
            + filename.rsplit(
                ".",
                1,
            )[1].lower()
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
    # PROCESS ONE IMAGE AT A TIME
    # ========================================================

    for image_index, uploaded_image in enumerate(
        images,
        start=1,
    ):

        filename = (
            uploaded_image.filename
            or f"image_{image_index}.jpg"
        )

        image_bytes: Optional[bytes] = None
        pil_image: Optional[Image.Image] = None
        temp_image_path: Optional[str] = None

        try:

            # =================================================
            # READ IMAGE
            # =================================================

            try:

                image_bytes = (
                    await uploaded_image.read()
                )

                if not image_bytes:

                    raise ValueError(
                        "Uploaded file is empty."
                    )

                # Prevent decompression bomb attacks
                # from consuming excessive RAM.
                Image.MAX_IMAGE_PIXELS = 25_000_000

                pil_image = Image.open(
                    io.BytesIO(
                        image_bytes
                    )
                ).convert(
                    "RGB"
                )

            except Exception as exc:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not read image "
                        f"'{filename}': {exc}"
                    ),
                ) from exc

            image_width, image_height = (
                pil_image.size
            )

            # =================================================
            # WRITE TEMPORARY FILE
            # =================================================

            suffix = (
                Path(filename).suffix.lower()
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

            # Release upload bytes as soon as
            # the temporary file has been written.
            del image_bytes
            image_bytes = None

            gc.collect()

            # =================================================
            # YOLO DETECTION
            # =================================================

            try:

                detector_output = detect_image(
                    temp_image_path
                )

            except Exception as exc:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"YOLO detection failed "
                        f"for '{filename}': {exc}"
                    ),
                ) from exc

            # =================================================
            # DELETE TEMP FILE
            # =================================================

            try:

                Path(
                    temp_image_path
                ).unlink(
                    missing_ok=True
                )

            except Exception:

                pass

            temp_image_path = None

            # =================================================
            # NORMALIZE DETECTIONS
            # =================================================

            detections = extract_detections(
                detector_output
            )

            # We no longer need the raw detector output.
            del detector_output

            # =================================================
            # COUNTS
            # =================================================

            image_counts = calculate_counts(
                detections
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

            # =================================================
            # WBC CLASSIFICATIONS
            # =================================================

            wbc_classifications: List[
                Dict[str, Any]
            ] = []

            # -------------------------------------------------
            # Process WBCs one at a time.
            # -------------------------------------------------

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

                # Only WBCs are sent to ConvNeXt.
                if class_name != WBC_CLASS_NAME:
                    continue

                bbox = normalize_bbox(
                    detection.get(
                        "bbox"
                    )
                )

                if bbox is None:
                    continue

                yolo_confidence = safe_float(
                    detection.get(
                        "confidence",
                        0.0,
                    )
                )

                # =================================================
                # CREATE WBC CROP
                # =================================================

                wbc_crop: Optional[
                    Image.Image
                ] = None

                try:

                    wbc_crop = crop_wbc(
                        pil_image,
                        bbox,
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

                    wbc_result = {
                        "detection_index": (
                            detection_index
                        ),

                        "yolo_confidence": round(
                            yolo_confidence,
                            6,
                        ),

                        "bbox": bbox,

                        "classification": (
                            failed_classification
                        ),

                        "crop_image": None,
                    }

                    wbc_classifications.append(
                        wbc_result
                    )

                    all_wbc_classifications.append(
                        {
                            "image_index": image_index,
                            "filename": filename,
                            **wbc_result,
                        }
                    )

                    continue

                # =================================================
                # CONVNEXT FOLD 2
                # =================================================

                try:

                    classification_output = (
                        classify_wbc_crop(
                            pil_image,
                            bbox,
                        )
                    )

                    classification_output = (
                        normalize_classification_output(
                            classification_output
                        )
                    )

                except Exception as exc:

                    classification_output = (
                        create_failed_classification(
                            "ConvNeXt Fold 2 "
                            "classification failed: "
                            f"{exc}"
                        )
                    )

                # =================================================
                # SMALL CROP PREVIEW
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

                    del crop_base64

                except Exception:

                    crop_image_data = None

                # =================================================
                # STORE RESULT
                # =================================================

                wbc_result: Dict[
                    str,
                    Any
                ] = {

                    "detection_index": (
                        detection_index
                    ),

                    "yolo_confidence": round(
                        yolo_confidence,
                        6,
                    ),

                    "bbox": bbox,

                    "classification": (
                        classification_output
                    ),

                    "crop_image": (
                        crop_image_data
                    ),
                }

                wbc_classifications.append(
                    wbc_result
                )

                all_wbc_classifications.append(
                    {
                        "image_index": image_index,
                        "filename": filename,
                        **wbc_result,
                    }
                )

                # =================================================
                # RELEASE CROP IMMEDIATELY
                # =================================================

                try:

                    wbc_crop.close()

                except Exception:

                    pass

                del wbc_crop

                gc.collect()

            # =================================================
            # IMAGE WBC SUMMARY
            # =================================================

            image_wbc_summary = (
                get_classification_summary(
                    wbc_classifications
                )
            )

            # =================================================
            # IMAGE RESULT
            # =================================================

            image_result: Dict[
                str,
                Any
            ] = {

                "index": image_index,

                "filename": filename,

                "content_type": (
                    uploaded_image.content_type
                    or "application/octet-stream"
                ),

                "image_size": {
                    "width": image_width,
                    "height": image_height,
                },

                "counts": image_counts,

                "detection_count": len(
                    detections
                ),

                "detections": detections,

                "wbc_classifications": (
                    wbc_classifications
                ),

                "wbc_subtype_analysis": (
                    image_wbc_summary
                ),
            }

            results.append(
                image_result
            )

            # =================================================
            # RELEASE TEMPORARY OBJECTS
            # =================================================

            del detections
            del wbc_classifications
            del image_result
            del image_wbc_summary
            del image_counts

            gc.collect()

        finally:

            # =================================================
            # CLEAN TEMP FILE
            # =================================================

            if temp_image_path is not None:

                try:

                    Path(
                        temp_image_path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:

                    pass

            # =================================================
            # CLOSE PIL IMAGE
            # =================================================

            if pil_image is not None:

                try:

                    pil_image.close()

                except Exception:

                    pass

                del pil_image

            # =================================================
            # RELEASE UPLOAD
            # =================================================

            try:

                await uploaded_image.close()

            except Exception:

                pass

            if image_bytes is not None:

                del image_bytes

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

    return {

        "status": "success",

        "message": (
            "Images analyzed successfully "
            "using BCCD YOLOv11 and "
            "ConvNeXt-Tiny Fold 2."
        ),

        "image_count": len(
            images
        ),

        "max_images": MAX_IMAGES,

        # ----------------------------------------------------
        # CELL COUNTS
        # ----------------------------------------------------

        "total_counts": total_counts,

        # ----------------------------------------------------
        # GLOBAL WBC ANALYSIS
        # ----------------------------------------------------

        "wbc_subtype_analysis": (
            global_wbc_summary
        ),

        # ----------------------------------------------------
        # IMAGE RESULTS
        # ----------------------------------------------------

        "results": results,

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        "inference": {

            "detector": {

                "model": (
                    "yolo11s_bccd_best.pt"
                ),

                "task": (
                    "BCCD blood-cell detection"
                ),

                "confidence_threshold": 0.25,

                "iou_threshold": 0.45,

                "device": "cpu",

                "image_size": 416,
            },

            "classifier": {

                "model": (
                    "ConvNeXt-Tiny"
                ),

                "ensemble": (
                    "single-fold"
                ),

                "folds": 1,

                "selected_fold": 2,

                "classes": 13,

                "input_size": "224x224",

                "feature_dimension": 768,

                "task": (
                    "WBC subtype classification"
                ),

                "decision_rule": (
                    "Prediction from Fold 2"
                ),

                "status": "completed",
            },

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            "pipeline": {

                "stage_1": (
                    "YOLOv11 BCCD detection"
                ),

                "stage_2": (
                    "WBC/RBC/Platelet counting"
                ),

                "stage_3": (
                    "WBC bounding-box extraction"
                ),

                "stage_4": (
                    "WBC crop generation"
                ),

                "stage_5": (
                    "ConvNeXt-Tiny Fold 2 inference"
                ),

                "stage_6": (
                    "JSON response generation"
                ),
            },

            # ------------------------------------------------
            # MEMORY OPTIMIZATION
            # ------------------------------------------------

            "memory_optimization": {

                "single_classifier_fold": True,

                "selected_fold": 2,

                "sequential_image_processing": True,

                "small_crop_previews": True,

                "crop_preview_size": (
                    MAX_CROP_PREVIEW_SIZE
                ),

                "crop_jpeg_quality": (
                    CROP_JPEG_QUALITY
                ),

                "explicit_garbage_collection": True,

                "temporary_files_removed": True,

                "device": "cpu",
            },
        },
    }