# 🚀 Performance Comparison of YOLOv5 & YOLOv8 on Curated and Augmented Datasets for Locust Detection  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14996738.svg)](https://doi.org/10.5281/zenodo.14996738)
🔍 A deep learning-based object detection model for locust identification using YOLOv5 and YOLOv8.


## 📌 Project Overview

This is the official repository for my research paper 
### "Enhanced Locust Detection in Smart Farming Using YOLOv5 and YOLOv8 with Data Augmentation: A Comparative Performance Evaluation" with DOI 10.48084/etasr.11843 . This study aims to detect locusts in images using YOLOv5 and YOLOv8 models. The dataset has been curated, annotated and augmentated specifically for this task.
This repository presents a comparative evaluation of YOLOv5 and YOLOv8 for locust detection. The study was conducted on:

1️⃣ A curated dataset specifically prepared for locust detection.

2️⃣ An augmented version of the dataset, where various augmentation techniques were applied to enhance model robustness.
   
3️⃣  Labels are in YOLO format. 

## 🔥 Key Findings:

◆ YOLOv8 outperformed YOLOv5 in both scenarios, achieving higher accuracy and better generalization.

◆ Dataset augmentation significantly improved model performance and provide higher F1 Score and better detection accuracy.


## ✅ Features

★ YOLOv5 & YOLOv8 implementations

★ Custom dataset (without augmentation and with augmentation) for locust detection

★ Training and inference scripts included

★ Supports Google Colab for easy execution



## 📁 Dataset & Model Files

🔹 Curated & Augmented Dataset are available on Zenodo. Dataset DOI: 10.5281/zenodo.14964987

🔹 Code Implementations: Includes training and inference scripts for both YOLOv5 and YOLOv8. 

🔹 Dataset is structured like this


```
locust_detection
├── images
│ ├── train
│ │ ├── img_001.jpg
│ │ ├── img_002.jpg
│ │ └── ...
│ ├── val
│ │ ├── img_101.jpg
│ │ ├── img_102.jpg
│ │ └── ...
└── labels
├── train
│ ├── img_001.txt
│ ├── img_002.txt
│ └── ...
├── val
│ ├── img_101.txt
│ ├── img_102.txt
│ └── ...
```

## 🎯 Augmentation Techniques Applied


Following augmentation techniques were applied to improve model robustness:

🔹 Flipped Horizontally

🔹 Flipped Vertically

🔹 Brightness & Contrast

🔹 Rotated & Scaled

🔹 Gaussian Blur

🔹 HSV Adjusted

🔹 Color Shifted

🔹 Cropped Locust

💡 Augmentation Script: Available in augmentation.ipynb

## 📸 Sample output of an image after augmentation 

### **Applied Augmentation Techniques**
![Augmentation](results/augmentation.jpg)



## 🚀 Installation & Setup


1️⃣ Clone the Repository

git clone https://github.com/yourusername/locust-detection.git
cd locust-detection

2️⃣ Install Dependencies

pip install -r requirements.txt

👉 If using Google Colab, run:

!pip install -r requirements.txt


3️⃣ Download the Dataset
Since the dataset is hosted on Zenodo, download it manually from:
🔗 Dataset DOI: 10.5281/zenodo.14964987

Unzip and place it in the correct directory (datasets/).

4️⃣ Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')



## 🔧 Training 

We trained and tested YOLOv5 and YOLOv8 on:

◆ Curated dataset (Original images)

◆ Augmented dataset (Images enhanced using transformations)

## ⚡ Results showed that YOLOv8 outperformed YOLOv5 in detection accuracy especially after augmentation.

✔ Train YOLOv5

!python train.py --img 640 --batch 16 --epochs 60 --data robo.dataset.yaml --weights yolov5s.pt --cache

✔ Train YOLOv8

results = model.train(

    data=dataset_yaml_path,  # Use the correct path here
    epochs=60,
    batch=16,
    imgsz=640,
    save_period=10,
    project=save_model_path,
    name='yolov8_detection'
)


## 🏆 Inference


✔ Inference YOLOv5

!python detect.py --weights /content/drive/MyDrive/locust_detection/yolov5/runs/train/exp/weights/last.pt \
                   --conf 0.40 --data dataset.yaml \
                   --source /content/drive/MyDrive/locust_detection/test_data/images

✔ Inference YOLOv8


results = model.predict(source=source_directory, conf=0.40, save=True, project=output_directory, name='detection_results')

## 📸 Visual Comparison of Results
Below are performance results:

### **YOLOv5 and YOLOv8 Performance on dataset without Augmentation**
![without augmentation](results/without.jpg)

### **YOLOv5 and YOLOv8 Performance on dataset with Augmentation**
![with augmentation](results/with.jpg)

## Below are sample detection results:

### **YOLOv5 Output after augmentation**
![YOLOv5 Detection](results/yolov5_output.jpg)

### **YOLOv8 Output after augmentation**
![YOLOv8 Detection](results/yolov8_output.jpg)


## 📜 Citation

If you use this dataset, please cite the Zenodo DOI:  
🔗 DOI: [10.5281/zenodo.14964987](https://doi.org/10.5281/zenodo.14964987)  

```bibtex
@misc{vajpayee2025locust,
  author    = {Pooja Vajpayee and Kuldeep Kr. Yogi},
  title     = {Locust Images Dataset},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14964987},
  url       = {https://doi.org/10.5281/zenodo.14964987}
}
```



## 📄 License
🔹 This project is licensed under the MIT License.

## 🤝 Acknowledgments
Special thanks to Ultralytics for YOLO development.

## 📬 Contact

For questions, reach out via poojavjpy@gmail.com or https://www.researchgate.net/profile/Pooja-Vajpayee-2/research

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14996738.svg)](https://doi.org/10.5281/zenodo.14996738)

