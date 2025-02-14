# MCU Model Deployment Demo

## Overview
This demo showcases the deployment of a custom YOLO-Fastest model on an STM32 microcontroller. The chosen custom model achieves inference speeds of approximately 2.5 FPS on the MCU.

## Demo Result
TBU

## Features
- **Real-time Inference:** Performs object detection on a live camera feed.
- **Optimized for MCU:** Uses YOLO-Fastest to ensure efficient inference within hardware constraints.
- **TFT LCD Display:** Outputs detection results directly to a screen for visualization.

## Requirements
- STM32 H74x/H75x MCU
- Pretrained YOLO-Fastest model
- Camera module OV767 or other module (for testing)
- LCD TFT ILI9341 (for testing)

## Others
This experimental demo illustrates the model's deployment on an MCU, capturing real-time camera input and rendering detections on a TFT LCD at approximately 10 FPS. However, this test version is not used in my thesis due to its lower mean Average Precision (mAP), which impacts detection accuracy.

https://github.com/user-attachments/assets/4007471b-7130-4bf8-93dc-eac93aaf5b90

---

© 2025 Nhat-Trieu Huynh-Pham
