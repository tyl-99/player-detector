# Football Player Tracking with yolov5 and DeepSort

This project demonstrates the use of YOLOv5m to detect football players in videos and highlight them with an ellipse drawn below each player, distinguishing players by their team colors.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview
![image](https://github.com/tyl-99/player-detector/assets/71328888/b9262f24-8640-4c9d-8cd3-191503f7e633)


This project aims to detect football players in video footage using the YOLOv5m model. The detected players are highlighted with an ellipse drawn below each player, and the players are separated by their team colors.

## Features

- **Player Detection**: Detect football players in video footage using YOLOv5m.
- **Team Color Separation**: Highlight players with an ellipse below each player, using their dominant team colors to distinguish between different teams.

## Model Architecture

The model used in this project is a combination of YOLOv5m for detection and DeepSort for tracking.

### YOLOv5m

YOLOv5m is part of the YOLO (You Only Look Once) family of models. It is a medium-sized version of the YOLOv5 model, designed for a balance between speed and accuracy.

- **Backbone**: CSPDarknet53, which extracts essential features from the input image.
- **Neck**: PANet (Path Aggregation Network), which aggregates features from different stages of the backbone.
- **Head**: Predicts bounding boxes, objectness scores, and class probabilities at three different scales.

### DeepSort

DeepSort (Deep Simple Online and Realtime Tracking) is used for tracking the detected players across video frames. It uses a combination of motion and appearance information to track objects reliably.

- **Kalman Filter**: For predicting the position of the tracked objects.
- **Hungarian Algorithm**: For data association between detections and tracklets.
- **Feature Extractor**: A pre-trained CNN that extracts appearance features for each detected object, helping to distinguish between different players.

### Combined Architecture

1. **Detection with YOLOv5m**: The YOLOv5m model detects players in each frame, providing bounding boxes and class probabilities.
2. **Tracking with DeepSort**: The DeepSort tracker assigns a unique ID to each detected player and tracks them across frames using motion and appearance information.
3. **Team Color Separation**: The detected players are highlighted with ellipses below each player, using their dominant team colors to distinguish between different teams.


### Advantages of YOLOv5m:

- **Speed**: YOLOv5m is designed to run efficiently on GPUs, making it suitable for real-time applications.
- **Accuracy**: The model achieves high accuracy in object detection tasks by leveraging advanced techniques like CSP connections and PANet.
- **Versatility**: YOLOv5m can be trained on a wide variety of datasets and is capable of detecting multiple types of objects simultaneously.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- CuPy

You can install the required packages using the following command:

```bash
pip install torch opencv-python numpy cupy-cuda
```

## Usage

To run the script and detect football players in a video, use the following command:

```bash
python PlayerTracking.py
```

Ensure that you have your YOLOv5 model weights and the input video file path correctly specified in the script.

