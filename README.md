
---
<img width="1614" height="696" alt="image" src="https://github.com/user-attachments/assets/a8518a37-0112-45a9-b28b-f97bbcca6c45" />

# DermalScan: AI Facial Skin Aging Detection App

DermalScan is an end-to-end **AI-based facial skin aging detection system** that analyzes facial images and classifies visible skin aging signs such as **wrinkles, dark spots, puffy eyes, and clear skin** using deep learning.
The project integrates **computer vision, transfer learning, and a web-based interface** to provide interpretable and user-friendly results.

---

## Project Objectives

* Detect facial skin aging signs from images
* Use a pretrained deep learning model for accurate classification
* Provide visual outputs with confidence percentages
* Build a complete pipeline from data preprocessing to deployment

---

## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **EfficientNetB0 (Transfer Learning)**
* **OpenCV (Haar Cascade for face detection)**
* **Streamlit (Web UI)**
* **NumPy, Pandas, Matplotlib**

---

## Dataset Description

* Dataset consists of **facial images categorized into four classes**:

  * Wrinkles
  * Dark Spots
  * Puffy Eyes
  * Clear Skin

* The uploaded dataset is **preprocessed and structured** into training and testing folders.

### Dataset Structure

```
preprocessed_dataset/
├── train/
│   ├── Wrinkles/
│   ├── clear skin/
│   ├── dark spots/
│   └── puffy eyes/
└── test/
    ├── Wrinkles/
    ├── clear skin/
    ├── dark spots/
    └── puffy eyes/
```

> Raw images are not included to avoid redundancy and licensing issues.

---

## Project Workflow

1. Image upload via Streamlit UI
2. Face detection using Haar Cascade
3. Image preprocessing (resize, normalization)
4. Feature extraction using EfficientNetB0
5. Classification into aging categories
6. Visualization with bounding boxes and probabilities
7. Export of annotated image and CSV logs

---

## Model Architecture

* **Base Model:** EfficientNetB0 (pretrained on ImageNet)

* **Input Size:** 224 × 224 × 3

* **Layers Added:**

  * Global Average Pooling
  * Dropout
  * Dense Softmax Output (4 classes)

* **Loss Function:** Categorical Cross-Entropy

* **Optimizer:** Adam

---

## Model Performance

* **Training Accuracy:** 95%
* **Best Validation Accuracy:** 88%    
* Strong predictions for visually prominent features
* Stable training and validation curves
* Inference time: **< 3 seconds per image**

---

## Web Application (Streamlit)

The Streamlit-based UI allows users to:

* Upload facial images
* View real-time predictions
* See annotated bounding boxes
* Analyze class-wise confidence scores
* Download results and logs

---

## Export & Logging (Module 7)

* Annotated image can be downloaded
* Predictions are logged in CSV format
* Each log includes:

  * Timestamp
  * Image name
  * Class probabilities
  * Number of detected faces

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

## 📁 Project Structure

```
DermalScan/
├── app.py
├── preprocessing.py
├── predict_pipeline.py
├── convert_model.py
├── dermalscan_streamlit.keras
├── preprocessed_dataset/
├── assets/
├── prediction_logs.csv
├── requirements.txt
└── README.md
```





