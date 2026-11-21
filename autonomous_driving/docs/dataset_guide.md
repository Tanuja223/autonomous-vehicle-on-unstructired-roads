# Dataset Preparation Guide

## 1. CARLA Synthetic Data Collection

```bash
python scripts/collect_carla_data.py --scenario DENSE_URBAN --frames 10000
```

## 2. GTSRB Traffic Signs
Download: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
Extract to: datasets/traffic_signs/

## 3. YOLO Fine-tuning
```bash
yolo train data=datasets/carla.yaml model=yolov8n.pt epochs=100 imgsz=640
```
