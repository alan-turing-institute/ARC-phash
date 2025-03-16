# AI Detection

This folder contains scripts for detecting AI-generated images using pre-trained models. Below is a description of each script and its usage.

## Scripts

### 1. `detector.py`

This file defines the `AIDetector` class and a custom image classification pipeline for detecting AI-generated images.

#### Classes

- `CustomImageClassificationPipeline`: A custom pipeline for image classification using a specified model.
- `AIDetector`: A class for detecting AI-generated images using a specified model.

#### Functions

- `load_models(model_str)`: Loads the specified model for image classification.

### 2. `run_detection.py`

This script uses the `AIDetector` class to detect AI-generated images in a specified directory.

#### Usage

```bash
python run_detection.py <model_name> <image_dir>
```

#### Arguments
- `model_name`: The name of the model to use for detection (e.g., Dafilab/ai-image-detector, Organika/sdxl-detector).
- `image_dir`: The directory containing the images to be processed.

