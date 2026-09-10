# 🩸 WBC Detection & Classification

> An end-to-end AI pipeline for detecting white blood cells from blood-smear images and classifying them into 13 white blood cell subtypes using YOLO-based detection, ConvNeXt-Tiny classification, synthetic-data verification, 3-fold ensemble inference, and uncertainty-aware analysis.

---

## 📌 Overview

This project implements an end-to-end computer vision pipeline for automated White Blood Cell (WBC) analysis from peripheral blood smear images.

Instead of directly classifying an entire blood-smear image, the system follows a two-stage approach:

1. **Detect cellular regions** using YOLO.
2. **Extract the detected WBC** and classify its subtype using an ensemble of ConvNeXt-Tiny models.

The final inference system combines predictions from three independently trained ConvNeXt-Tiny folds and evaluates prediction reliability using multiple uncertainty indicators.

### High-Level Pipeline

```text
                 BLOOD SMEAR IMAGE
                        │
                        ▼
                   YOLO Detector
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
            WBC        RBC     Platelets
             │
             ▼
        WBC Detection
             │
             ▼
      WBC Crop + 15% Padding
             │
             ▼
     ┌───────┼────────┐
     ▼       ▼        ▼
   Fold 1   Fold 2   Fold 3
  ConvNeXt ConvNeXt ConvNeXt
     │       │        │
     └───────┼────────┘
             ▼
    Probability Ensemble
             │
             ▼
        WBC Subtype
             │
     ┌───────┼───────────────┐
     ▼       ▼               ▼
Confidence Entropy        Margin
     │
     ├── Fold Agreement
     │
     └── TTA Agreement
             │
             ▼
   Reliability / Uncertainty
             │
             ▼
      Streamlit Dashboard
````

---

# 🎯 Objectives

The main objectives of the project are:

-  Detect WBCs from blood-smear images. 
-  Isolate individual WBC regions from complex smear backgrounds. 
-  Classify WBCs into 13 morphological subtypes. 
-  Address severe class imbalance in the original dataset. 
-  Increase training diversity using synthetic and augmented images. 
-  Automatically verify generated synthetic images. 
-  Reduce dependence on a single classifier through 3-fold ensemble inference. 
-  Evaluate model behavior using class-wise metrics and confusion matrices. 
-  Provide model interpretability through Grad-CAM. 
-  Estimate prediction reliability using multiple uncertainty indicators. 
-  Deploy the complete inference workflow through an interactive Streamlit dashboard. 

---

# 🧬 WBC Classes

The classification system contains **13 WBC classes**:

| Code | WBC Type |
| ------------ | -------------------- |
| BA           | Basophil             |
| BL           | Blast                |
| BNE          | Band-form Neutrophil |
| EO           | Eosinophil           |
| LY           | Lymphocyte           |
| MMY          | Metamyelocyte        |
| MO           | Monocyte             |
| MY           | Myelocyte            |
| PC           | Plasma Cell          |
| PLY          | Prolymphocyte        |
| PMY          | Promyelocyte         |
| SNE          | Segmented Neutrophil |
| VLY          | Variant Lymphocyte   |

These classes are represented throughout the original WBCBench dataset, the synthetic-data pipeline, and the final ConvNeXt classification system.

---

# 📊 Dataset

## WBCBench Dataset

The project is based on the **WBCBench dataset**, which is organized into two major phases.

The dataset structure contains:

```
```

```
white-blood-cells/
│
├── phase1/
│
├── phase2/
│   ├── train/
│   ├── eval/
│   └── test/
│
├── phase1_label.csv
├── phase2_train.csv
├── phase2_eval.csv
└── phase2_test.csv
```

The dataset split described in the project consists of:

| Split | Patients | Images | Purpose |
| -------------------------- | --- | ------ | ------------------------------------ |
| Phase 1                    | 74  | 8,288  | Pristine training data               |
| Phase 2 Train              | 222 | 24,897 | Training data with image degradation |
| Phase 2 Eval               | 49  | 5,350  | Evaluation                           |
| Phase 2 Test               | 148 | 16,477 | Final test / leaderboard inference   |

The dataset therefore provides both relatively clean images and degraded images, making it useful for evaluating robustness under different imaging conditions. 

---

# 📈 Original Class Distribution

The original training data consists of **Phase 1 + Phase 2 Train**.

The distribution is highly imbalanced.

Approximate original image counts shown in the project analysis include:

| Class | Original Images |
| -------------------- | ------ |
| SNE                  | 15,048 |
| LY                   | 9,230  |
| MO                   | 3,195  |
| BL                   | 2,271  |
| EO                   | 979    |
| MY                   | 513    |
| BA                   | 494    |
| BNE                  | 451    |
| MMY                  | 433    |
| VLY                  | 382    |
| PMY                  | 118    |
| PC                   | 58     |
| PLY                  | 13     |

This imbalance is particularly severe for classes such as PLY, PC, PMY, and BNE.

The class-distribution analysis is shown in the project documentation on page 4. 

---

# 🧪 Dataset Construction Pipeline

The final classification dataset was not simply the raw WBCBench training set.

The project uses a multi-stage data preparation strategy.

```
```

```
WBCBench Dataset
       │
       ▼
Phase 1 + Phase 2 Training Data
       │
       ▼
Image Preprocessing
       │
       ├── Resize
       ├── RGB Conversion
       └── Label Encoding
       │
       ▼
Class Distribution Analysis
       │
       ▼
Synthetic Data Generation
       │
       ├── MedFusion
       └── Classical Augmentation
       │
       ▼
Synthetic Image Verification
       │
       ▼
Verified Synthetic Dataset
       │
       ▼
Merge with Original Images
       │
       ▼
Final Training Dataset
```

The complete data-construction workflow is documented in the project pipeline. 

---

# 🧠 Synthetic Data Generation

To address class imbalance, synthetic WBC images were generated for underrepresented classes.

The project uses **MedFusion**, a diffusion-based image generation approach.

The MedFusion architecture is based on a U-Net diffusion model with:

-  Time embedding 
-  Residual blocks 
-  Downsampling blocks 
-  Bottleneck 
-  Upsampling blocks 
-  Skip connections 
-  Self-attention 

The diffusion model learns to predict noise added to WBC images and uses the reverse diffusion process to generate realistic WBC images.

---

# 🔬 MedFusion Training

The MedFusion training process follows:

```
```

```
Real WBC Image
      │
      ▼
Sample Random Timestep
      │
      ▼
Sample Gaussian Noise
      │
      ▼
Noise Schedule
      │
      ▼
Forward Diffusion
      │
      ▼
Noisy WBC Image
      │
      ▼
MedFusion U-Net
      │
      ▼
Predicted Noise
      │
      ▼
MSE Loss
      │
      ▼
Backpropagation
      │
      ▼
AdamW Optimization
```

The project documentation describes the diffusion training process using random timesteps, Gaussian noise, a noise schedule, forward diffusion, predicted-noise estimation, MSE loss, and AdamW optimization. 

---

# 🧬 Synthetic Classes

Synthetic generation was primarily used to increase representation of minority WBC classes.

Examples shown in the project include:

-  Promyelocyte (PMY) 
-  Plasma Cell (PC) 
-  Prolymphocyte (PLY) 

The generated images were then subjected to an automated verification stage before being included in the final training data.

---

# 🔍 Synthetic Image Verification

Generated images are **not automatically accepted**.

A trained YOLO detector is used to verify whether a generated image contains a valid WBC.

The verification pipeline evaluates:

-  Number of detected cells 
-  Detection confidence 
-  Cell size 
-  Bounding-box characteristics 
-  Cell position 
-  Aspect ratio / morphology 

The project rejects generated images when:

1.  No WBC is detected. 
2.  Multiple WBCs are detected. 
3.  Detection confidence is too low. 
4.  Cell size is outside the accepted range. 
5.  The detected cell is incorrectly positioned. 
6.  The cell morphology/aspect ratio is considered unrealistic. 

Only images satisfying the defined quality criteria are retained. 

---

# 📦 Synthetic Verification Results

Examples from the verification process include:

### PC

```
```

```
Processed: 1000
Accepted:   456
Rejected:   544
```

### PLY

```
```

```
Processed: 975
Accepted:  597
Rejected:  378
```

### PMY

```
```

```
Processed: 1000
Accepted:  623
Rejected:  377
```

The PMY verification result is documented on page 28. 

The PC and PLY verification results are also shown in the project documentation. 

---

# 🔄 Classical Image Augmentation

In addition to synthetic generation, classical image augmentation was used to increase visual diversity.

The augmentation pipeline includes:

-  Horizontal Flip 
-  Vertical Flip 
-  Rotation 
-  Shift / Scale / Rotation 
-  Random Brightness / Contrast 
-  CLAHE 
-  Image normalization 

The project documentation specifies rotation ranges and brightness/contrast adjustments as part of the augmentation strategy. 

These augmentations are intended to improve robustness against:

-  Orientation changes 
-  Position changes 
-  Scale variations 
-  Illumination differences 
-  Staining variations 
-  Low-contrast regions 

---

# ⚖️ Final Training Dataset

After synthetic-image verification and augmentation, the accepted synthetic samples were merged with the original training images.

The final class distribution was substantially more balanced for the minority classes.

The final dataset distribution documented in the project is:

| Class | Final Training Images |
| -------------------------- | ------ |
| BA                         | 3,000  |
| BL                         | 3,000  |
| BNE                        | 3,000  |
| EO                         | 3,000  |
| LY                         | 9,230  |
| MMY                        | 3,000  |
| MO                         | 3,195  |
| MY                         | 3,000  |
| PC                         | 2,514  |
| PLY                        | 2,610  |
| PMY                        | 2,741  |
| SNE                        | 15,048 |
| VLY                        | 3,000  |

The final class distribution is shown in the project's final balanced-training-data analysis. 

> Note: The repository should treat the generated/augmented training images as derived data rather than replacing the original WBCBench dataset.

---

# 🚀 YOLO Detection

The first stage of the final inference pipeline is WBC detection.

```
```

```
Blood Smear
    │
    ▼
YOLO Detector
    │
    ├── WBC
    ├── RBC
    └── Platelets
```

The detector identifies cellular regions in the complete blood-smear image.

The WBC detection result is then used to isolate the WBC for the classification stage.

---

# 🧪 YOLO Training & Verification

The project documentation describes a dedicated YOLO pipeline for:

1.  Loading WBC images. 
2.  Automatic ROI detection. 
3.  Generating bounding boxes. 
4.  Converting annotations into YOLO format. 
5.  Creating train/validation datasets. 
6.  Training the YOLO detector. 
7.  Selecting the best checkpoint. 
8.  Running detection on generated images. 
9.  Extracting detection statistics. 
10.  Applying class-specific verification thresholds. 
11.  Accepting or rejecting synthetic images. 

The documented synthetic-image verification workflow is shown on page 25. 

### Model used during documented synthetic-data verification

```
```

```
YOLOv11n
```

This is the model explicitly documented in the training/verification presentation. 

### Current deployed detector

The current final application pipeline uses:

```
```

```
YOLOv11s
```

This distinction is intentional: the documented synthetic-data verification stage and the current deployed inference detector are separate project stages.

---

# ✂️ WBC Crop Extraction

Once a WBC is detected:

```
```

```
Detected Bounding Box
        │
        ▼
Bounding Box Expansion
        │
        ▼
15% Padding
        │
        ▼
WBC Crop
        │
        ▼
ConvNeXt Classifier
```

A **15% padding margin** is added around the detected WBC before classification.

This provides additional contextual information around the cell while keeping the WBC as the dominant visual region.

---

# 🧠 ConvNeXt-Tiny Classification

The extracted WBC crop is classified using **ConvNeXt-Tiny**.

The model uses transfer learning with the final classification layer adapted for the project's **13 WBC classes**.

```
```

```
WBC Crop
   │
   ▼
Resize → 224 × 224
   │
   ▼
RGB Image
   │
   ▼
Pretrained ConvNeXt-Tiny
   │
   ▼
Classification Head
   │
   ▼
13-Class Probability Vector
```

---

# 🏋️ ConvNeXt Training Configuration

The documented training configuration includes:

| ParameterValue    |                      |
| ----------------- | -------------------- |
| Architecture      | ConvNeXt-Tiny        |
| Number of Classes | 13                   |
| Image Size        | 224 × 224            |
| Batch Size        | 32                   |
| Epochs            | 10                   |
| Learning Rate     | 3 × 10⁻⁴             |
| Weight Decay      | 1 × 10⁻⁴             |
| Loss              | Cross Entropy Loss   |
| Optimizer         | AdamW                |
| LR Scheduler      | Cosine Annealing     |
| Training Strategy | Transfer Learning    |
| Validation        | Stratified 3-Fold CV |
| Mixed Precision   | AMP                  |

The complete training workflow and these hyperparameters are shown in the project training pipeline. 

---

# 🔀 Stratified 3-Fold Cross-Validation

Instead of relying on a single train/validation split, the project trains three independent ConvNeXt-Tiny models.

```
```

```
Final Training Dataset
        │
        ▼
Stratified 3-Fold Split
        │
   ┌────┼────┐
   ▼    ▼    ▼
 Fold1 Fold2 Fold3
   │    │    │
   ▼    ▼    ▼
Model1 Model2 Model3
   │    │    │
   └────┼────┘
        ▼
 Probability Ensemble
```

The training workflow reports approximately:

```
```

```
Training images:   37,558
Validation images:18,780
```

with slightly different train/validation counts for each fold due to stratified splitting. 

---

# 🧮 Training Process

Each fold follows the same training procedure:

```
```

```
Training Dataset
      │
      ▼
PyTorch DataLoader
      │
      ▼
Augmentation / Normalization
      │
      ▼
ConvNeXt-Tiny
      │
      ▼
Forward Pass
      │
      ▼
Cross Entropy Loss
      │
      ▼
Backpropagation
      │
      ▼
AdamW
      │
      ▼
Cosine Annealing LR
      │
      ▼
Validation
      │
      ▼
Best Fold Checkpoint
```

Mixed precision training using AMP is incorporated into the training pipeline. 

---

# 📊 ConvNeXt Fold Results

## Fold 1

| Metric | Score |
| ----------------- | ------ |
| Accuracy          | 0.8950 |
| Balanced Accuracy | 0.8612 |
| Macro Precision   | 0.8694 |
| Macro Recall      | 0.8612 |
| Macro F1          | 0.8624 |

The Fold-1 confusion matrix and classification report are documented in the project results. 

---

## Fold 2

| Metric | Score |
| ----------------- | ------ |
| Accuracy          | 0.9431 |
| Balanced Accuracy | 0.9316 |
| Macro Precision   | 0.9332 |
| Macro Recall      | 0.9316 |
| Macro F1          | 0.9320 |

The Fold-2 classification report and confusion matrix are included in the project documentation. 

---

## Fold 3

| Metric | Score |
| ----------------- | ------ |
| Accuracy          | 0.9622 |
| Balanced Accuracy | 0.9557 |
| Macro Precision   | 0.9619 |
| Macro Recall      | 0.9557 |
| Macro F1          | 0.9586 |

The Fold-3 classification report and confusion matrix are documented on page 38. 

---

# 📈 Cross-Fold Summary

| Fold | Accuracy | Balanced Acc. | Macro Precision | Macro Recall | Macro F1 |
| ------------------------------------------------------------ | ------ | ------ | ------ | ------ | ------ |
| Fold 1                                                       | 89.50% | 86.12% | 86.94% | 86.12% | 86.24% |
| Fold 2                                                       | 94.31% | 93.16% | 93.32% | 93.16% | 93.20% |
| Fold 3                                                       | 96.22% | 95.57% | 96.19% | 95.57% | 95.86% |

The three-fold results demonstrate progressively stronger performance across the reported folds.

---

# 🔎 Model Interpretability — Grad-CAM

Grad-CAM was used to investigate which image regions influenced ConvNeXt predictions.

```
```

```
WBC Image
    │
    ▼
ConvNeXt
    │
    ▼
Target Class
    │
    ▼
Grad-CAM
    │
    ▼
Activation Heatmap
```

The Grad-CAM visualizations show that the classifier generally focuses on the WBC region rather than relying exclusively on surrounding RBC structures.

Grad-CAM analyses were generated for all three folds.   

---

# 🧪 Phase-2 Evaluation

The trained fold checkpoints were additionally evaluated on the **5,350 Phase-2 evaluation images**.

This provides a more challenging evaluation setting because Phase 2 contains degraded images.

The project evaluates each fold independently and reports:

-  Accuracy 
-  Balanced Accuracy 
-  Macro Precision 
-  Macro Recall 
-  Macro F1 
-  Confusion Matrix 

The evaluation results are documented in the project presentation. 

### Reported results

| Fold | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |
| ---------------------------------------------------------------- | ------ | ------ | ------ | ------ | ------ |
| Fold 1                                                           | 90.37% | 54.41% | 65.31% | 54.41% | 54.61% |
| Fold 2                                                           | 92.93% | 64.20% | 73.03% | 64.20% | 67.20% |
| Fold 3                                                           | 92.60% | 61.38% | 72.20% | 61.38% | 64.79% |

The Phase-2 evaluation results reveal an important distinction between overall accuracy and class-balanced performance, highlighting the effect of distribution shift and class imbalance under degraded conditions. 

---

# 🧩 Probability Ensemble

For final inference, the three ConvNeXt fold models are combined.

Instead of using a majority vote, the system combines the predicted probability distributions.

For an input image:

```
```

```
Fold 1 → P₁(class)
Fold 2 → P₂(class)
Fold 3 → P₃(class)
```

The ensemble probability is:

```
```

```
Pensemble =
(P₁ + P₂ + P₃) / 3
```

The final WBC subtype is selected from the highest ensemble probability.

```
```

```
          Fold 1
             │
             ▼
        Probability
             │
             ├─────────┐
             │         │
          Fold 2    Fold 3
             │         │
             └────┬────┘
                  ▼
         Probability Average
                  │
                  ▼
           Final Prediction
```

---

# 🎯 Reliability & Uncertainty Analysis

The system goes beyond simply returning:

```
```

```
Prediction = "LY"
Confidence = 95%
```

Instead, it evaluates multiple indicators of prediction reliability.

The current pipeline considers:

### 1. Confidence

The highest ensemble class probability.

```
```

```
confidence = max(Pensemble)
```

---

### 2. Prediction Margin

Difference between the highest and second-highest class probabilities.

```
```

```
margin =
P(top-1) - P(top-2)
```

A larger margin indicates a clearer separation between the predicted class and the runner-up.

---

### 3. Entropy

Entropy measures how distributed the prediction probability is across classes.

```
```

```
H(P) = -Σ Pᵢ log(Pᵢ)
```

Lower entropy generally indicates a more concentrated prediction distribution.

Higher entropy indicates greater uncertainty.

---

### 4. Fold Agreement

The three independently trained models are compared.

```
```

```
Fold 1 → LY
Fold 2 → LY
Fold 3 → LY

Agreement = High
```

versus:

```
```

```
Fold 1 → LY
Fold 2 → MO
Fold 3 → LY

Agreement = Lower
```

This provides an additional signal about model stability.

---

### 5. TTA Agreement

Test-Time Augmentation (TTA) can be used to evaluate whether predictions remain stable under small input transformations.

Stable predictions across TTA variants indicate greater robustness.

---

# 🛡️ Reliability-Aware Prediction

The final system therefore considers multiple signals:

```
```

```
             Prediction
                 │
     ┌───────────┼────────────┐
     ▼           ▼            ▼
 Confidence   Entropy       Margin
     │           │            │
     └───────────┼────────────┘
                 ▼
        Fold Agreement
                 │
                 ▼
         TTA Agreement
                 │
                 ▼
        Reliability Score
                 │
                 ▼
        Dashboard Output
```

This makes the system more suitable for analysis-oriented workflows than a simple single-model classifier.

---

# 🧪 Phase-2 Test Inference

The final Phase-2 test set contains:

```
```

```
16,477 images
```

The trained models were used to generate predictions for the test set.

The project records the predicted class distribution.

Example output:

| Class | Predictions |
| ---------------- | ----- |
| BA               | 269   |
| BL               | 907   |
| BNE              | 102   |
| EO               | 482   |
| LY               | 4,295 |
| MMY              | 176   |
| MO               | 1,654 |
| MY               | 178   |
| PC               | 6     |
| PMY              | 34    |
| SNE              | 8,172 |
| VLY              | 202   |

The project documentation reports these values as the predicted distribution on the Phase-2 test dataset. 

> Because the Phase-2 test labels are not available for local evaluation, this distribution should **not** be interpreted as test-set accuracy.

---

# 🖥️ Streamlit Dashboard

The complete inference pipeline is exposed through an interactive **Streamlit dashboard**.

The dashboard allows a user to provide a blood-smear image and obtain an end-to-end analysis.

### Dashboard Workflow

```
```

```
Upload Blood Smear Image
          │
          ▼
      YOLO Detection
          │
          ▼
      Detect WBC
          │
          ▼
    Crop + 15% Padding
          │
          ▼
   3 ConvNeXt Models
          │
          ▼
 Probability Ensemble
          │
          ▼
    WBC Classification
          │
          ▼
 Reliability Analysis
          │
          ▼
      Dashboard
```

---

# 📊 Dashboard Output

The dashboard is designed to expose model outputs such as:

-  Input blood-smear image 
-  Detection results 
-  WBC crop 
-  Predicted WBC subtype 
-  Ensemble probabilities 
-  Confidence 
-  Prediction margin 
-  Entropy 
-  Fold agreement 
-  TTA agreement 
-  Reliability / uncertainty information 

The purpose is to make the model's decision process more transparent and easier to inspect.

---

# 🧰 Technology Stack

## Machine Learning

-  Python 
-  PyTorch 
-  ConvNeXt-Tiny 
-  Transfer Learning 
-  Mixed Precision / AMP 
-  AdamW 
-  Cross Entropy Loss 
-  Cosine Annealing Learning Rate Scheduler 

## Object Detection

-  YOLOv11 
-  Ultralytics 
-  OpenCV 
-  Bounding-box based WBC extraction 

## Generative AI

-  Diffusion Models 
-  U-Net 
-  MedFusion 
-  Gaussian Noise / Forward Diffusion 
-  Reverse Diffusion 

## Computer Vision

-  OpenCV 
-  PIL 
-  Image normalization 
-  CLAHE 
-  Geometric augmentation 
-  Brightness / contrast augmentation 

## Model Evaluation

-  Scikit-learn 
-  Accuracy 
-  Balanced Accuracy 
-  Precision 
-  Recall 
-  Macro F1 
-  Confusion Matrix 
-  Grad-CAM 

## Deployment / UI

-  Streamlit 
-  Python 

## Development

-  Git 
-  GitHub 
-  VS Code 
-  Virtual Environment 

---

# 📁 Project Structure

```
```

```
wbc-detection-classification/
│
├── .devcontainer/
│
├── backend/
│   │
│   ├── app/
│   │   └── ...
│   │
│   ├── models/
│   │   └── ...
│   │
│   ├── weights/
│   │   └── ...
│   │
│   ├── Dockerfile
│   ├── streamlit_app.py
│   └── test_detector.py
│
├── frontend/
│   └── ...
│
├── .gitattributes
├── .gitignore
├── packages.txt
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## 1. Clone the repository

```
```

```
git clone https://github.com/Parthiv511/wbc-detection-classification.git
cd wbc-detection-classification
```

---

## 2. Create a virtual environment

### Windows

```
```

```
python -m venv .venv
```

Activate it:

```
```

```
.venv\Scripts\activate
```

### Linux / macOS

```
```

```
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```
```

```
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

From the project root:

```
```

```
streamlit run backend/streamlit_app.py
```

The application should open in the browser.

---

# ☁️ Deployment

The application is designed to be deployed using **Streamlit Community Cloud**.

The deployment entry point is:

```
```

```
backend/streamlit_app.py
```

The root-level `requirements.txt` contains the dependencies required by the Streamlit application.

System-level Linux packages required by the application are specified in:

```
```

```
packages.txt
```

---

# ⚙️ Configuration

The deployed application requires the trained model weights.

Typical model assets include:

```
```

```
backend/
├── weights/
│   ├── detector weights
│   ├── ConvNeXt fold 1
│   ├── ConvNeXt fold 2
│   └── ConvNeXt fold 3
```

Large model files should be managed appropriately rather than unnecessarily committed to Git history.

---

# 🧪 Testing

The repository includes detector testing utilities.

Example:

```
```

```
python backend/test_detector.py
```

The test workflow can be used to verify that the detection stage loads correctly and produces expected WBC detections.

---

# 🔬 Reproducibility

The classification pipeline uses:

-  Fixed class mappings 
-  Consistent preprocessing 
-  Defined image size 
-  Stratified 3-fold validation 
-  Saved fold checkpoints 
-  Deterministic dataset construction where applicable 
-  Explicit training hyperparameters 

This makes it possible to reproduce the classification experiments from the documented pipeline.

---

# 📌 Key Engineering Decisions

## Why a two-stage architecture?

Direct classification of a full blood-smear image introduces irrelevant visual information.

The two-stage approach separates:

```
```

```
Localization
     +
Classification
```

YOLO identifies the WBC region, while ConvNeXt focuses on morphological classification.

---

## Why ConvNeXt-Tiny?

ConvNeXt-Tiny provides a modern convolutional architecture suitable for high-resolution visual feature extraction while remaining practical for transfer learning.

---

## Why 3-fold ensemble?

A single model may have fold-specific biases.

Three independently trained models provide:

-  Multiple learned representations 
-  More stable predictions 
-  Probability-level ensemble predictions 
-  Fold agreement as an additional reliability signal 

---

## Why synthetic data?

The original dataset contains substantial class imbalance.

Synthetic data generation was therefore used to improve representation of underrepresented WBC classes.

However, generated images were subjected to automated YOLO-based verification before being accepted.

---

## Why uncertainty analysis?

A classifier's maximum probability alone does not completely describe prediction reliability.

The project therefore considers:

```
```

```
Confidence
+
Margin
+
Entropy
+
Fold Agreement
+
TTA Agreement
```

This provides a richer picture of model stability.

---

# 📊 Evaluation Strategy

The project uses multiple levels of evaluation.

```
```

```
Level 1
│
├── Fold-wise validation
│
├── Accuracy
├── Balanced Accuracy
├── Macro Precision
├── Macro Recall
├── Macro F1
└── Confusion Matrix

Level 2
│
├── Phase-2 Evaluation
├── Degraded Images
└── Fold-wise robustness analysis

Level 3
│
├── Phase-2 Test Inference
└── Prediction distribution

Level 4
│
├── Grad-CAM
└── Model interpretability

Level 5
│
├── Confidence
├── Entropy
├── Margin
├── Fold Agreement
└── TTA Agreement
```

---

# ⚠️ Limitations

This project is intended as an AI research/engineering system and **not as a standalone medical diagnostic system**.

Important limitations include:

-  Dataset distribution may not represent all laboratories or imaging devices. 
-  Phase-2 degradation introduces distribution shift. 
-  Some WBC classes remain visually similar. 
-  Class imbalance can still affect performance. 
-  Synthetic images may introduce artifacts. 
-  Phase-2 test labels are not available for local evaluation. 
-  Model confidence should not be interpreted as clinical certainty. 
-  Deployment performance may depend on available CPU/GPU resources. 

---

# 🔮 Future Improvements

Potential future work includes:

-  Calibrated confidence estimation 
-  Temperature scaling 
-  More advanced uncertainty estimation 
-  OOD detection 
-  Better synthetic-image quality filtering 
-  More robust augmentation strategies 
-  Model quantization 
-  ONNX/TensorRT inference 
-  GPU-optimized deployment 
-  Automated model monitoring 
-  Experiment tracking 
-  CI/CD for model testing 
-  Automated regression tests for inference 
-  API-based production deployment 
-  More comprehensive explainability 
-  External validation on independent datasets 

---

# 🏆 Project Highlights

### Dataset Engineering

-  WBCBench Phase-1 and Phase-2 data 
-  13 WBC classes 
-  Severe class imbalance analysis 
-  Synthetic data generation 
-  Automated synthetic-image verification 
-  Classical augmentation 
-  Final balanced training dataset 

### Deep Learning

-  YOLO-based WBC detection 
-  ConvNeXt-Tiny classification 
-  Transfer learning 
-  Stratified 3-fold cross-validation 
-  Probability-level ensemble 
-  Mixed precision training 

### Model Evaluation

-  Accuracy 
-  Balanced Accuracy 
-  Macro Precision 
-  Macro Recall 
-  Macro F1 
-  Confusion matrices 
-  Phase-2 robustness evaluation 
-  Grad-CAM interpretability 

### Reliability

-  Confidence 
-  Prediction margin 
-  Entropy 
-  Fold agreement 
-  TTA agreement 
-  Reliability-aware inference 

### Deployment

-  Streamlit dashboard 
-  End-to-end image inference 
-  Interactive prediction analysis 
-  Cloud deployment ready 

---

# 📜 End-to-End Summary

```
```

```
                         WBCBench
                            │
             ┌──────────────┴──────────────┐
             │                             │
          Phase 1                      Phase 2
        8,288 images             Train / Eval / Test
             │                             │
             └──────────────┬──────────────┘
                            ▼
                    Dataset Analysis
                            │
                            ▼
                    Class Imbalance
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Classical                  MedFusion
       Augmentation               Generation
              │                           │
              │                    Synthetic Images
              │                           │
              │                           ▼
              │                    YOLO Verification
              │                           │
              │                    Accepted Images
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 Final Training Dataset
                            │
                            ▼
                Stratified 3-Fold CV
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
         Fold 1           Fold 2           Fold 3
       ConvNeXt-Tiny    ConvNeXt-Tiny    ConvNeXt-Tiny
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                    Best Checkpoints
                            │
                            ▼
                  YOLO WBC Detection
                            │
                            ▼
                  WBC Crop + Padding
                            │
                            ▼
                 3-Fold ConvNeXt Ensemble
                            │
                            ▼
                     WBC Subtype
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
          Confidence     Entropy        Margin
              │
              ├──────── Fold Agreement
              │
              └──────── TTA Agreement
                            │
                            ▼
                  Reliability Analysis
                            │
                            ▼
                   Streamlit Dashboard
```

---

# 👨‍💻 Author

**Parthiv511**

GitHub:

https\://github.com/Parthiv511

---

# 📄 Disclaimer

This project is developed for educational, research, and engineering purposes.

It is not intended to replace professional hematological examination, laboratory testing, or clinical diagnosis.
