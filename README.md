# Football Player Detector

This project demonstrates the use of YOLOv5m to detect football players in videos and highlight them with an ellipse drawn below each player, distinguishing players by their team colors.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview
![image](https://github.com/tyl-99/player-detector/assets/71328888/84f6903d-fcc7-4d9f-8a18-9c0b983b358b)

This project aims to detect football players in video footage using the YOLOv5m model. The detected players are highlighted with an ellipse drawn below each player, and the players are separated by their team colors.

## Features

- **Player Detection**: Detect football players in video footage using YOLOv5m.
- **Team Color Separation**: Highlight players with an ellipse below each player, using their dominant team colors to distinguish between different teams.

## Model Architecture

The model used in this project is YOLOv5m, which is part of the YOLO (You Only Look Once) family of models. YOLOv5m is a medium-sized version of the YOLOv5 model, designed for a balance between speed and accuracy.

### Key Components of YOLOv5m:

- **Backbone**: The backbone is responsible for extracting essential features from the input image. YOLOv5m uses CSPDarknet53 as its backbone, which is a variation of Darknet53 enhanced with Cross Stage Partial (CSP) connections to improve learning and reduce computational cost.
- **Neck**: The neck of the model aggregates features from different stages of the backbone to construct feature pyramids. YOLOv5m uses PANet (Path Aggregation Network) for this purpose, which helps in enhancing the feature maps for better object detection.
- **Head**: The head of the model is responsible for predicting the bounding boxes, objectness scores, and class probabilities. YOLOv5m uses an anchor-based approach, predicting bounding boxes at three different scales to detect objects of various sizes.

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
python football_player_detector.py
```

Ensure that you have your YOLOv5 model weights and the input video file path correctly specified in the script.

