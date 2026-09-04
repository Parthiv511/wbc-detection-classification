from pathlib import Path
from typing import Any, Dict, List
import gc

import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "yolo11s_bccd_best.pt"


# ============================================================
# MODEL
# ============================================================

# The model is loaded only when detection is requested.
# This prevents unnecessary model loading during FastAPI startup.
model = None


def get_model():
    """
    Load the YOLO model only when it is first required.

    This keeps FastAPI startup lightweight.
    """

    global model

    if model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found:\n{MODEL_PATH}"
            )

        print("[Detector] Loading YOLO model...")
        print(f"[Detector] Model path: {MODEL_PATH}")

        model = YOLO(str(MODEL_PATH))

        print("[Detector] YOLO model loaded successfully.")
        print(f"[Detector] Model classes: {model.names}")

    return model


def release_model():
    """
    Release the YOLO model from RAM.

    This is important for Render's 512 MB memory limit.

    The application uses YOLO for detection first and
    ConvNeXt for classification afterward. Keeping both
    models in memory simultaneously can exceed the limit.
    """

    global model

    if model is not None:

        print("[Detector] Releasing YOLO model from memory...")

        model = None

        # Force Python garbage collection.
        gc.collect()

        # CPU tensors do not require CUDA cleanup.
        # This check is kept for portability.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Detector] YOLO model released.")


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "WBC",
    1: "RBC",
    2: "Platelets",
}

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45


# ============================================================
# MEMORY OPTIMIZATION SETTINGS
# ============================================================

# Smaller inference resolution reduces YOLO activation memory.
#
# 416 is a good starting point for a 512 MB environment.
# If detection quality becomes noticeably worse, we can test
# 512 later.
YOLO_IMAGE_SIZE = 416


# Limit the maximum number of detections returned.
#
# This prevents very large detection tensors from consuming
# unnecessary memory.
YOLO_MAX_DETECTIONS = 100


# ============================================================
# RUN YOLO DETECTION
# ============================================================

def detect_image(
    image_path: str,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> Dict[str, Any]:

    model_instance = get_model()

    detections: List[Dict[str, Any]] = []

    counts = {
        "WBC": 0,
        "RBC": 0,
        "Platelets": 0,
    }

    results = None

    try:

        print("[Detector] Starting YOLO inference...")
        print(f"[Detector] Image: {image_path}")
        print(f"[Detector] Image size: {YOLO_IMAGE_SIZE}")
        print(f"[Detector] Confidence: {conf}")
        print(f"[Detector] IoU: {iou}")

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------
        #
        # inference_mode prevents PyTorch from creating
        # autograd information, reducing memory usage.
        #
        with torch.inference_mode():

            results = model_instance.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                imgsz=YOLO_IMAGE_SIZE,
                max_det=YOLO_MAX_DETECTIONS,
                device="cpu",
                verbose=False,
            )

        print("[Detector] YOLO inference completed.")

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        for result in results:

            boxes = getattr(result, "boxes", None)

            if boxes is None:
                continue

            for box in boxes:

                # --------------------------------------------
                # Class ID
                # --------------------------------------------

                class_id = int(
                    box.cls[0].item()
                )

                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = float(
                    box.conf[0].item()
                )

                # --------------------------------------------
                # Class name
                # --------------------------------------------

                if class_id in CLASS_NAMES:

                    class_name = CLASS_NAMES[class_id]

                else:

                    # Safely handle an unexpected class ID.
                    try:
                        class_name = str(
                            model_instance.names[class_id]
                        )
                    except (KeyError, IndexError):
                        class_name = str(class_id)

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                coordinates = box.xyxy[0].tolist()

                x1, y1, x2, y2 = coordinates

                # --------------------------------------------
                # Count classes
                # --------------------------------------------

                if class_name in counts:

                    counts[class_name] += 1

                # --------------------------------------------
                # Store detection
                # --------------------------------------------

                detections.append(
                    {
                        "class_id": class_id,

                        "class_name": class_name,

                        "confidence": round(
                            confidence,
                            4,
                        ),

                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                    }
                )

        print(
            f"[Detector] Detection complete. "
            f"Total detections: {len(detections)}"
        )

        print(
            f"[Detector] Counts: {counts}"
        )

        return {
            "counts": counts,
            "detections": detections,
        }

    finally:

        # ----------------------------------------------------
        # CRITICAL MEMORY CLEANUP
        # ----------------------------------------------------
        #
        # We must remove the YOLO results before releasing
        # the model because result objects can hold references
        # to tensors.
        # ----------------------------------------------------

        if results is not None:

            del results

        # Remove the local model reference.
        model_instance = None

        # Release the global YOLO model.
        release_model()

        # Force garbage collection.
        gc.collect()

        print("[Detector] Detection memory cleanup completed.")


# ============================================================
# CROP ONE WBC
# ============================================================

def crop_wbc(
    image_path: str,
    bbox: Dict[str, float],
    padding: float = 0.15,
) -> Image.Image:

    image = Image.open(
        image_path
    ).convert("RGB")

    image_width, image_height = image.size

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
    # Calculate padding
    # --------------------------------------------------------

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = box_width * padding
    pad_y = box_height * padding

    # --------------------------------------------------------
    # Apply padding
    # --------------------------------------------------------

    x1 -= pad_x
    y1 -= pad_y

    x2 += pad_x
    y2 += pad_y

    # --------------------------------------------------------
    # Keep coordinates inside image
    # --------------------------------------------------------

    x1 = max(
        0,
        min(
            x1,
            image_width,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            image_height,
        ),
    )

    x2 = max(
        0,
        min(
            x2,
            image_width,
        ),
    )

    y2 = max(
        0,
        min(
            y2,
            image_height,
        ),
    )

    # --------------------------------------------------------
    # Convert to integer coordinates
    # --------------------------------------------------------

    crop_box = (
        int(x1),
        int(y1),
        int(x2),
        int(y2),
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    return image.crop(
        crop_box
    )


# ============================================================
# CROP ALL DETECTED WBCs
# ============================================================

def crop_detected_wbcs(
    image_path: str,
    detections: List[Dict[str, Any]],
    padding: float = 0.15,
) -> List[Dict[str, Any]]:

    wbc_crops = []

    wbc_index = 1

    for detection in detections:

        # Only crop WBC detections.
        if detection["class_name"] != "WBC":
            continue

        bbox = detection["bbox"]

        crop = crop_wbc(
            image_path=image_path,
            bbox=bbox,
            padding=padding,
        )

        wbc_crops.append(
            {
                "wbc_index": wbc_index,

                "confidence": detection[
                    "confidence"
                ],

                "bbox": bbox,

                "crop": crop,
            }
        )

        wbc_index += 1

    return wbc_crops


# ============================================================
# COMPLETE DETECTION + WBC CROPPING
# ============================================================

def detect_and_crop_wbcs(
    image_path: str,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    padding: float = 0.15,
) -> Dict[str, Any]:

    detection_result = detect_image(
        image_path=image_path,
        conf=conf,
        iou=iou,
    )

    wbc_crops = crop_detected_wbcs(
        image_path=image_path,
        detections=detection_result[
            "detections"
        ],
        padding=padding,
    )

    return {
        "counts": detection_result[
            "counts"
        ],

        "detections": detection_result[
            "detections"
        ],

        "wbc_crops": wbc_crops,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict[str, Any]:

    model_instance = get_model()

    try:

        return {
            "model": MODEL_PATH.name,

            "path": str(
                MODEL_PATH
            ),

            "classes": model_instance.names,

            "task": "BCCD blood-cell detection",

            "confidence_threshold":
                DEFAULT_CONFIDENCE,

            "iou_threshold":
                DEFAULT_IOU,

            "image_size":
                YOLO_IMAGE_SIZE,

            "max_detections":
                YOLO_MAX_DETECTIONS,
        }

    finally:

        # /model-info should not leave YOLO in RAM.
        release_model()