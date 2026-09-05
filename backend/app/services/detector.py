from pathlib import Path
from typing import Any, Dict, List

import gc

from PIL import Image
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "yolo11s_bccd_best.pt"
)


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

# ------------------------------------------------------------
# Render 512 MB optimization
# ------------------------------------------------------------

IMAGE_SIZE = 416
MAX_DETECTIONS = 100


# ============================================================
# MODEL
# ============================================================

model = None


def get_model():
    """
    Load YOLO only when required.

    The model is loaded once and reused.
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

    return model


# ============================================================
# DETECTION
# ============================================================

def detect_image(
    image_path: str,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> Dict[str, Any]:
    """
    Run YOLO detection on one image.

    Memory optimized for Render 512 MB:
    - CPU inference
    - imgsz=416
    - max_det=100
    - Convert YOLO tensors immediately to Python values
    - Delete YOLO results after processing
    """

    model_instance = get_model()

    print("[Detector] Running YOLO detection...")

    results = model_instance.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    counts = {
        "WBC": 0,
        "RBC": 0,
        "Platelets": 0,
    }

    # --------------------------------------------------------
    # Convert YOLO results into plain Python objects
    # --------------------------------------------------------

    for result in results:

        boxes = getattr(result, "boxes", None)

        if boxes is None:
            continue

        for box in boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = CLASS_NAMES.get(
                class_id,
                str(
                    model_instance.names[class_id]
                ),
            )

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1, y1, x2, y2 = coordinates

            if class_name in counts:
                counts[class_name] += 1

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

    # --------------------------------------------------------
    # Release YOLO prediction results immediately
    # --------------------------------------------------------

    del results
    gc.collect()

    print(
        f"[Detector] Detection complete. "
        f"WBC={counts['WBC']}, "
        f"RBC={counts['RBC']}, "
        f"Platelets={counts['Platelets']}"
    )

    return {
        "counts": counts,
        "detections": detections,
    }


# ============================================================
# CROP ONE WBC
# ============================================================

def crop_wbc(
    image: Image.Image,
    bbox: Dict[str, float],
    padding: float = 0.15,
) -> Image.Image:
    """
    Crop one WBC from an already opened PIL image.

    Important:
    The image is passed in rather than opened repeatedly.
    This reduces memory and file-I/O overhead.
    """

    image_width, image_height = image.size

    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])

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
        min(x1, image_width),
    )

    y1 = max(
        0,
        min(y1, image_height),
    )

    x2 = max(
        0,
        min(x2, image_width),
    )

    y2 = max(
        0,
        min(y2, image_height),
    )

    crop_box = (
        int(x1),
        int(y1),
        int(x2),
        int(y2),
    )

    return image.crop(crop_box)


# ============================================================
# CROP ALL DETECTED WBCs
# ============================================================

def crop_detected_wbcs(
    image_path: str,
    detections: List[Dict[str, Any]],
    padding: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Crop all detected WBCs.

    Memory optimization:
    - Open the original image only ONCE.
    - Reuse the same PIL image for all crops.
    - Close the original image after all crops are created.
    """

    wbc_crops: List[Dict[str, Any]] = []

    wbc_index = 1

    # --------------------------------------------------------
    # Open image only once
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    try:

        for detection in detections:

            if detection["class_name"] != "WBC":
                continue

            bbox = detection["bbox"]

            crop = crop_wbc(
                image=image,
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

    finally:

        # ----------------------------------------------------
        # Close original image
        # ----------------------------------------------------

        image.close()

    gc.collect()

    print(
        f"[Detector] Created "
        f"{len(wbc_crops)} WBC crop(s)."
    )

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
    """
    Complete pipeline:

        1. Run YOLO detection
        2. Extract detections
        3. Count WBC/RBC/Platelets
        4. Crop detected WBCs
        5. Return plain detection data + WBC crops
    """

    print(
        "[Detector] Starting detection + WBC cropping..."
    )

    # --------------------------------------------------------
    # STEP 1: Detection
    # --------------------------------------------------------

    detection_result = detect_image(
        image_path=image_path,
        conf=conf,
        iou=iou,
    )

    # --------------------------------------------------------
    # STEP 2: Crop WBCs
    # --------------------------------------------------------

    wbc_crops = crop_detected_wbcs(
        image_path=image_path,
        detections=detection_result["detections"],
        padding=padding,
    )

    # --------------------------------------------------------
    # STEP 3: Return result
    # --------------------------------------------------------

    result = {
        "counts": detection_result["counts"],

        "detections": detection_result["detections"],

        "wbc_crops": wbc_crops,
    }

    gc.collect()

    print(
        "[Detector] Detection + cropping complete."
    )

    return result


# ============================================================
# UNLOAD YOLO MODEL
# ============================================================

def unload_model():
    """
    Explicitly release the YOLO model.

    Useful for Streamlit / low-memory environments.
    """

    global model

    if model is not None:

        print("[Detector] Unloading YOLO model...")

        try:
            model.cpu()
        except Exception:
            pass

        try:
            del model
        except Exception:
            pass

        model = None

    gc.collect()

    print(
        "[Detector] YOLO memory released."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict[str, Any]:
    """
    Return model configuration without running inference.
    """

    model_instance = get_model()

    return {
        "model": MODEL_PATH.name,

        "path": str(MODEL_PATH),

        "classes": model_instance.names,

        "task": "BCCD blood-cell detection",

        "confidence_threshold":
            DEFAULT_CONFIDENCE,

        "iou_threshold":
            DEFAULT_IOU,

        "image_size":
            IMAGE_SIZE,

        "max_detections":
            MAX_DETECTIONS,

        "device":
            "cpu",
    }