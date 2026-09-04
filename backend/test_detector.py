from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\Parthiv kumar\OneDrive\2\August-2026\WBC\wbc\backend\models\yolo11s_bccd_best.pt"
)

IMAGE_PATH = Path(
    r"C:\Users\Parthiv kumar\OneDrive\2\August-2026\WBC\BloodImage_00000.jpg"
)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print("Classes:", model.names)


# ============================================================
# RUN DETECTION
# ============================================================

print("\nRunning detection...")

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    iou=0.45,
    verbose=False,
    save=True,
)


# ============================================================
# PROCESS RESULTS
# ============================================================

for result in results:

    print("\n================================")
    print("YOLO DETECTION RESULTS")
    print("================================")

    # Use getattr so Pylance doesn't incorrectly complain
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        print("No cells detected.")
        continue

    counts = {
        "WBC": 0,
        "RBC": 0,
        "Platelets": 0,
    }

    for box in boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Get class name directly from the loaded model
        class_name = model.names[class_id]

        coordinates = box.xyxy[0].tolist()

        counts[class_name] += 1

        print(
            f"{class_name:12s} "
            f"confidence={confidence:.3f} "
            f"bbox={[round(x, 1) for x in coordinates]}"
        )


    # ========================================================
    # COUNTS
    # ========================================================

    print("\n================================")
    print("CELL COUNTS")
    print("================================")

    print(f"WBC        : {counts['WBC']}")
    print(f"RBC        : {counts['RBC']}")
    print(f"Platelets  : {counts['Platelets']}")

    print("\nAnnotated image has been saved by YOLO.")