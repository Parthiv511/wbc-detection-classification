from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "yolo11s_bccd_best.pt"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print(f"Model path: {MODEL_PATH}")
print(f"Model classes: {model.names}")


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
# RUN YOLO DETECTION
# ============================================================

def detect_image(
    image_path: str,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> Dict[str, Any]:

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    counts = {
        "WBC": 0,
        "RBC": 0,
        "Platelets": 0,
    }

    for result in results:

        boxes = getattr(result, "boxes", None)

        if boxes is None:
            continue

        for box in boxes:

            class_id = int(box.cls[0].item())

            confidence = float(
                box.conf[0].item()
            )

            class_name = CLASS_NAMES.get(
                class_id,
                str(model.names[class_id])
            )

            coordinates = (
                box.xyxy[0]
                .tolist()
            )

            x1, y1, x2, y2 = coordinates

            if class_name in counts:
                counts[class_name] += 1

            detections.append({

                "class_id": class_id,

                "class_name": class_name,

                "confidence": round(
                    confidence,
                    4
                ),

                "bbox": {

                    "x1": round(
                        x1,
                        2
                    ),

                    "y1": round(
                        y1,
                        2
                    ),

                    "x2": round(
                        x2,
                        2
                    ),

                    "y2": round(
                        y2,
                        2
                    ),

                },

            })

    return {

        "counts": counts,

        "detections": detections,

    }


# ============================================================
# CROP ONE WBC
# ============================================================

def crop_wbc(
    image_path: str,
    bbox: Dict[str, float],
    padding: float = 0.15,
) -> Image.Image:

    """
    Crop a detected WBC from the original blood-smear image.

    bbox format:
    {
        "x1": ...,
        "y1": ...,
        "x2": ...,
        "y2": ...
    }

    padding:
        Extra area around the WBC as a fraction of
        the bounding-box width/height.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

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

    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 + pad_x
    y2 = y2 + pad_y

    # --------------------------------------------------------
    # Keep coordinates inside image
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, image_width)
    )

    y1 = max(
        0,
        min(y1, image_height)
    )

    x2 = max(
        0,
        min(x2, image_width)
    )

    y2 = max(
        0,
        min(y2, image_height)
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

    cropped_image = image.crop(
        crop_box
    )

    return cropped_image


# ============================================================
# CROP ALL DETECTED WBCs
# ============================================================

def crop_detected_wbcs(
    image_path: str,
    detections: List[Dict[str, Any]],
    padding: float = 0.15,
) -> List[Dict[str, Any]]:

    """
    Generate crops for every WBC detected by YOLO.

    Only detections whose class_name == "WBC"
    are cropped.
    """

    wbc_crops = []

    wbc_index = 1

    for detection in detections:

        if detection["class_name"] != "WBC":
            continue

        bbox = detection["bbox"]

        crop = crop_wbc(
            image_path=image_path,
            bbox=bbox,
            padding=padding,
        )

        wbc_crops.append({

            "wbc_index": wbc_index,

            "confidence": detection[
                "confidence"
            ],

            "bbox": bbox,

            "crop": crop,

        })

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

    """
    Run YOLO detection and generate WBC crops.
    """

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
# SIMPLE MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict[str, Any]:

    return {

        "model": MODEL_PATH.name,

        "path": str(MODEL_PATH),

        "classes": model.names,

        "task": "BCCD blood-cell detection",

        "confidence_threshold":
            DEFAULT_CONFIDENCE,

        "iou_threshold":
            DEFAULT_IOU,

    }