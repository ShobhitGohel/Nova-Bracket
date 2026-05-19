Nova Bracket – AI-Based Bracket Placement Detection
Overview

Nova Bracket is an AI-powered dental bracket placement detection system developed using Computer Vision and YOLO object detection models. The project was designed to assist orthodontic analysis by detecting and identifying dental bracket positions from dental images in real time.

The complete workflow involved:

Dataset collection and annotation
Dataset preprocessing using Roboflow
YOLO model training on Google Colab
Real-time Android application integration
Technologies Used
Python
YOLOv11 / YOLOv26
Roboflow
LabelImg
Google Colab
Android Studio
Firebase (optional integration)
Computer Vision
Dataset Preparation
Image Collection
Collected approximately 8000 dental images for training and testing.
Images included multiple dental angles, lighting conditions, and patient cases.
Image Annotation

The dataset was manually labelled using:

Roboflow Labeller
LabelImg

Each dental bracket was annotated with bounding boxes for object detection training.

Dataset Processing

Using Roboflow:

Images were resized and normalized
Dataset versions were managed
Data augmentation techniques were applied:
Rotation
Brightness adjustment
Flipping
Scaling

This helped improve model generalization and real-world performance.

Model Training
Training Environment
Model training was performed using Google Colab with GPU acceleration.
YOLO Models Used
YOLOv11
YOLOv26
Training Workflow
Exported processed dataset from Roboflow in YOLO format
Uploaded dataset to Google Colab
Configured YOLO training environment
Trained detection models using Python
Evaluated model performance using validation metrics
Performance
Dataset Size: ~8000 images
Real-world Accuracy: ~85%
Successfully tested across multiple dental image conditions
Android Integration
Android Studio Deployment

The trained YOLO detection model was integrated into an Android application prototype using Android Studio.

Features
Real-time bracket detection
Camera-based image input
Detection visualization using bounding boxes
Mobile-friendly inference workflow

The Android application was developed to assist practical orthodontic analysis and testing.

Real-World Testing
The system has been tested and utilized by a dental student on approximately 50–60 patients.
Practical testing helped evaluate detection consistency under real clinical conditions.
Research Work

The project is currently under process for:

Research paper publication
Further model optimization
Expanded clinical testing
Future Improvements
Improve detection accuracy using larger datasets
Add multi-class dental analysis
Optimize mobile inference speed
Deploy cloud-based prediction APIs
Expand Android application features
Project Workflow Summary
Collect Dental Images
Annotate Images using Roboflow & LabelImg
Process & Augment Dataset in Roboflow
Export YOLO Dataset
Train YOLO Models in Google Colab
Evaluate Accuracy & Performance
Integrate Trained Model into Android Studio
Perform Real-World Clinical Testing
