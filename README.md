# 🦷 Nova Bracket — AI-Based Dental Bracket Placement Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v26-red?style=for-the-badge)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset%20Workflow-orange?style=for-the-badge)
![Android](https://img.shields.io/badge/Android-Studio-green?style=for-the-badge\&logo=android)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-AI-purple?style=for-the-badge)

</div>

---

# 📌 Overview

Nova Bracket is an AI-powered dental bracket placement detection system developed using **YOLO-based Computer Vision models**.
The project focuses on assisting orthodontic analysis by detecting dental bracket positions from dental images with real-time prediction capabilities.

The complete pipeline includes:

* Manual dataset annotation
* Dataset preprocessing & augmentation
* YOLO model training on Google Colab
* Android Studio integration for real-time detection

---

# 🚀 Features

✅ AI-based dental bracket detection
✅ Real-time object detection workflow
✅ Android application integration
✅ Custom annotated dataset
✅ Roboflow-powered preprocessing pipeline
✅ Practical clinical testing on real patients

---

# 🛠️ Tech Stack

| Category             | Technologies                |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| AI Models            | YOLOv11, YOLOv26            |
| Dataset Workflow     | Roboflow                    |
| Annotation Tools     | Roboflow Labeller, LabelImg |
| Training Environment | Google Colab                |
| Mobile Integration   | Android Studio              |
| Domain               | Computer Vision, AI         |

---

# 📂 Dataset Preparation

## 📸 Image Collection

* Collected and processed approximately **8000 dental images**
* Included multiple dental conditions, patient cases, and lighting environments

---

## 🏷️ Annotation Process

Images were manually labelled using:

* **Roboflow Labeller**
* **LabelImg**

Bounding boxes were created around dental brackets for supervised object detection training.

---

## 🔄 Dataset Processing with Roboflow

Roboflow was used for:

* Dataset version management
* Image preprocessing
* Dataset organization
* Data augmentation

### Applied Augmentations

* Rotation
* Brightness adjustment
* Scaling
* Flipping

This significantly improved model generalization during real-world testing.

---

# 🧠 Model Training

## ⚡ Training Environment

Training was performed using **Google Colab GPU instances**.

---

## 🤖 Models Used

* YOLOv11
* YOLOv26

---

## 📈 Training Workflow

```text
Dental Images
      ↓
Image Annotation
      ↓
Roboflow Dataset Processing
      ↓
YOLO Dataset Export
      ↓
Google Colab Training
      ↓
Model Evaluation
      ↓
Android Integration
```

---

# 📊 Model Performance

| Metric              | Result                   |
| ------------------- | ------------------------ |
| Dataset Size        | ~8000 Images             |
| Model Accuracy      | ~85% Real-World Accuracy |
| Framework           | YOLOv11 / YOLOv26        |
| Testing Environment | Clinical Dental Cases    |

---

# 📱 Android Integration

The trained YOLO model was integrated into an Android application prototype using **Android Studio**.

## Features

* Real-time bracket detection
* Camera-based image input
* Bounding box visualization
* Mobile inference workflow

The Android application was designed for practical orthodontic assistance and testing.

---

# 🏥 Real-World Usage

The project has been:

* Tested and utilized by a dental student
* Used across approximately **50–60 patient cases**
* Evaluated under practical clinical conditions

---

# 📄 Research Work

📌 The project is currently under process for:

* Research paper publication
* Further model optimization
* Expanded clinical validation

---

# 🔮 Future Improvements

* Increase dataset size for improved accuracy
* Add multi-class dental analysis
* Improve mobile inference speed
* Cloud-based prediction APIs
* Advanced orthodontic analytics

---

# 👨‍💻 Author

## Shobhit Gohel

AI | Android | Computer Vision | Data Analytics

* LinkedIn
* GitHub

---

# ⭐ Project Status

🚧 Active Development & Research Phase
