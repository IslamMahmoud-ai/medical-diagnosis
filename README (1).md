# 🩺 Medical Diagnosis AI System

## Overview
Deep learning system for multi-class disease classification from medical images using ResNet50 transfer learning, achieving **94% accuracy**.

## Tech Stack
- **Python** · **TensorFlow/Keras** · **OpenCV** · **Flask**

## Features
- ResNet50 transfer learning with custom classification head
- Data augmentation pipeline (rotation, zoom, flip)
- Multi-class disease classification (Normal, Pneumonia, Tumor, Fracture, Other)
- REST API for real-time inference
- Web dashboard for result visualization

## Project Structure
```
1_medical_diagnosis/
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── app.py
├── model/
│   └── medical_model.h5
├── data/
│   ├── train/
│   └── val/
└── Medical_Diagnosis_AI.ipynb
```

## Quick Start
```bash
pip install -r requirements.txt
python train.py --data_dir ./data --epochs 20
python app.py
# POST http://localhost:5000/predict  with image file
```

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 94% |
| Architecture | ResNet50 |
| Input Size | 224×224 |
| Classes | 5 |
