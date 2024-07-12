# EfficientDet-SWA: Sliding Window Attention for Object Detection

EfficientDet-SWA enhances object detection by integrating the EfficientDet model with a sliding window approach. This combination improves accuracy for smaller objects by analyzing image segments independently and refining results with non-maximum suppression (NMS). The project is modular and easy to extend.

## Introduction
EfficientDet-SWA leverages the EfficientDet model for object detection while enhancing detection capabilities using a sliding window approach. This method helps in identifying smaller objects that might be missed when processing the entire image at once.

## Features
- **EfficientDet Model**: Uses TensorFlow's EfficientDet model for high accuracy and efficiency.
- **Sliding Window Attention**: Implements a sliding window mechanism to enhance object detection.
- **Non-Maximum Suppression (NMS)**: Filters overlapping boxes to reduce redundancy. 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MrSaeidSeyfi/EfficientDet-SWA.git
   cd EfficientDet-SWA
   python main.py
   ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

