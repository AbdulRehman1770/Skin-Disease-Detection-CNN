# Skin Diseases Detection Using Deep Learning (4 Classes)

This repository contains a Deep Learning project designed to detect and classify skin diseases into four distinct categories using a custom-built Convolutional Neural Network (CNN) with TensorFlow/Keras. The project features an end-to-end pipeline from data preprocessing to an interactive user interface for testing predictions.

---

## 🎯 Problem Statement & Motivation (Targeting Rural Healthcare)
In many developing and remote **rural areas**, access to healthcare is severely limited. A critical shortage of professional dermatologists means that common and potentially severe skin conditions often go undiagnosed or misdiagnosed until they reach advanced stages. 

Furthermore, water contamination in rural communities frequently leads to high incidences of specific conditions like **Arsenicosis**. Traveling to urban diagnostic centers is both physically and financially exhausting for these underserved populations.

### The Solution:
This project aims to bridge this healthcare gap by providing an automated, accessible, and fast computer-vision-based screening tool. By leveraging deep learning, it can serve as a preliminary diagnostic aid in local health clinics, enabling early-stage detection even in the absence of a resident specialist.

---

## 🚀 Project Overview
This system implements a robust deep learning pipeline that processes skin images and classifies them into one of the following four target categories, focusing heavily on conditions prevalent in both everyday life and affected rural regions:
1. **Arsenicosis (Infected)** - Highly relevant to rural areas with groundwater chemical risks.
2. **Healthy Skin (Not Infected)**
3. **Acne**
4. **Eczema**

---

## 🧠 Neural Network & Model Architecture

Instead of relying on heavy pre-trained transfer learning models, this project utilizes a custom, deeply stacked **Convolutional Neural Network (CNN)** configured using the Keras Sequential API. This lightweight custom architecture ensures that the features unique to these specific skin lesions are captured effectively without excessive computational overhead.

### Model Layers & Structural Workflow:
* **Input Layer:** Standardizes all incoming multi-class image arrays to a resolution of `(224, 224, 3)`.
* **Convolutional Layers (`Conv2D`):** Extracts structural edges, shapes, and complex pixel texture patterns unique to different skin anomalies using `ReLU` activation.
* **Pooling Layers (`MaxPooling2D`):** Downsamples the spatial dimensions of feature maps while retaining essential activation information.
* **Regularization (`BatchNormalization` & `Dropout`):** Prevents overfitting and ensures stable convergence across training epochs.
* **Dense Layers:** Fully connected layers flatten spatial patterns into logical dense structures.
* **Output Layer:** A final fully connected layer with **4 units** utilizing the **Softmax activation function** to generate probability distributions for the 4 categories.

### Compilation Configs:
* **Optimizer:** Adam (Adaptive Moment Estimation)
* **Loss Function:** `categorical_crossentropy` (suited for multi-class targets)
* **Metrics:** Accuracy

---

## 📊 Dataset Distribution
The network processes data distributed among original and augmented streams partitioned across the target classes:
* **Arsenicosis / Infected Skin**
* **Healthy / Not Infected Skin**
* **Acne**
* **Eczema**

*Note: Data augmentation strategies (such as random rotations, shifts, and zooms) are configured to ensure class balance and structural generalization.*

---

## 🖥️ Interactive Evaluation UI
To make this tool accessible and easy to test for non-technical field workers or users, the project includes a practical interface built right inside the notebook using **Jupyter Widgets (`ipywidgets`)**. Users can utilize the graphical **"🚀 Analyze Images"** framework to dynamically upload testing samples, trigger evaluation modules, and visualize model predictions seamlessly within the environment.

---

## 💻 Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Language:** Python 3
* **Libraries:** NumPy, OpenCV (`cv2`), Matplotlib, OS, ipywidgets
* **Hardware Acceleration:** Configured for Google Colab T4 GPU environments

---

