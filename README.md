# H. Pylori Detection Using FCOS

This repository contains an anchor-free object detection pipeline based on FCOS (Fully Convolutional One-Stage detector) to automatically detect *Helicobacter pylori* (*H. pylori*) in H&E-stained histopathology images.

## [Dataset](https://drive.google.com/file/d/1YmxBuiQfq-r_CLf_Lk9CDrbR2NYMfQUn/view)
The dataset used for this project contains high-resolution H&E-stained histopathology images with bounding box annotations for *H. pylori* organisms.

### Key Characteristics:
- **Total Size**: 2,382 high-resolution source images.
- **Composition**: 
  - **1,931** positive images (containing *H. pylori*).
  - **376** negative images (background tissue only, no bacteria).
  - **75** dedicated held-out test images.
- **Resolutions**: Mixed high resolutions including 1920 x 1080, 2688 x 1512, and 3840 x 2160.
- **Object Size**: Individual *H. pylori* organisms are extremely small, spanning only 20-60 pixels across these massive images.
- **Annotations**: Bounding box coordinates provided in YOLO format (`x_center, y_center, width, height` normalized).

## Pipeline Overview

Because the original images are high resolution (up to 3840x2160) and the bacteria are very small (20-60 pixels), directly feeding the images into a CNN is computationally prohibitive and ineffective. We use a patchification strategy to solve this.

The pipeline consists of three main steps:

1. **Format Conversion (`convert_yolo_to_pascal.py`)**: 
   Converts the original YOLO format annotations (normalized `x_center, y_center, width, height`) to absolute Pascal VOC format (`xmin, ymin, xmax, ymax`).

2. **Patchification (`patchify.py`)**:
   Divides the high-resolution images into 512 x 512 patches with a stride of 256 (50% overlap). The overlapping ensures that bacteria on the boundary of one patch are fully visible in the adjacent patch. Bounding boxes are remapped to local patch coordinates.

3. **Model Architecture & Training (`h_pylori_fcos.ipynb`)**:
   We utilize **FCOS (Fully Convolutional One-Stage Object Detection)**, an anchor-free detector, which provides significant advantages for this task:
   - **Pretrained Backbone**: Uses a ResNet-50 backbone pre-trained on ImageNet for strong low-level feature extraction.
   - **Feature Pyramid Network (FPN)**: Multi-scale FPN (levels P3-P7) helps handle organisms across a range of sizes.
   - **Anchor-Free Shared Heads**: Eliminates the need for anchor box hyperparameter tuning. Instead, it predicts a classification score, 4 bounding-box regression distances, and a centerness score (to down-weight low-quality predictions) at every spatial location using detection heads that are shared across all FPN levels. This dense per-pixel prediction is highly effective for detecting small, irregularly shaped bacteria.
   - **Two-Phase Transfer Learning**: The ResNet-50 backbone is initially frozen to preserve pretrained features while the detection head learns to localize the bacteria. Afterwards, the entire model is fine-tuned with differential learning rates.
   - **Data Augmentation**: Extensive domain-specific augmentations (HSV jitter for stain variation, rotations, blurring) are applied using Albumentations to combat overfitting.
   - **Mixed Precision Training**: Uses `torch.cuda.amp` (with TF32 enabled) for accelerated training on Ampere/Turing GPUs.

## Usage

### 1. Requirements
Install the required packages:
```bash
pip install torch torchvision numpy pandas matplotlib albumentations scikit-learn torchmetrics Pillow
```

### 2. Prepare the Data
First, convert the YOLO annotations to Pascal VOC format:
```bash
python3 convert_yolo_to_pascal.py
```
*(Make sure to update the directory paths inside the script to point to your dataset location)*

Then, generate the patches:
```bash
python3 patchify.py
```

### 3. Training
Open and run `h_pylori_fcos.ipynb` on an environment with a GPU (e.g., Kaggle, Google Colab, or a local machine). Ensure the paths in the notebook correctly point to the generated patches directory.

## Results
The proposed pipeline achieved the following metrics on a held-out validation set (including a dedicated test set):
- **mAP@0.5**: 0.527
- **mAP@[0.5:0.95]**: 0.183
- **mAP@0.75**: 0.062

The model successfully detects the presence of *H. pylori*, though precise bounding box localization remains a challenge due to the imprecise, soft boundaries of the bacteria in histopathology tissue.
