# Food Spoilage Classification Using MobileNetV2

## 📌 Project Overview

Food spoilage is a major concern in the food industry, leading to economic losses and health risks. This project presents a deep learning-based Food Spoilage Classification System that automatically identifies fresh and rotten fruits and vegetables using computer vision and transfer learning.

The model is built using MobileNetV2, a lightweight and efficient Convolutional Neural Network (CNN) architecture, fine-tuned on a dataset containing multiple categories of fresh and spoiled produce. The system can accurately classify images into their respective freshness categories, making it suitable for smart agriculture, food quality monitoring, inventory management, and automated inspection systems.

---

## 🚀 Features

* Deep Learning-based image classification
* Transfer Learning using MobileNetV2
* Classification of fresh and rotten fruits and vegetables
* Fine-tuned model for improved performance
* High test accuracy of **96.45%**
* TensorFlow and Keras implementation
* Google Colab compatible

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

---

## 📂 Dataset

The dataset consists of images belonging to the following classes:

### Fresh Categories

* Fresh Apples
* Fresh Banana
* Fresh Bitter Gourd
* Fresh Capsicum
* Fresh Cucumber
* Fresh Okra
* Fresh Oranges
* Fresh Potato
* Fresh Tomato

### Rotten Categories

* Rotten Apples
* Rotten Banana
* Rotten Bitter Gourd
* Rotten Capsicum
* Rotten Cucumber
* Rotten Okra
* Rotten Oranges
* Rotten Potato
* Rotten Tomato

Total Classes: **18**

> Note: The dataset is not included in this repository due to its large size.

---

## 🧠 Model Architecture

### Base Model

* MobileNetV2 (Pre-trained on ImageNet)

### Custom Classification Head

* GlobalAveragePooling2D
* Dense Layer
* Dropout Layer
* Output Dense Layer (18 Classes)

### Input Shape

* 224 × 224 × 3 RGB Images

---

## 📊 Model Performance

| Metric            | Value                    |
| ----------------- | ------------------------ |
| Model             | MobileNetV2 (Fine-Tuned) |
| Input Size        | 224 × 224 × 3            |
| Number of Classes | 18                       |
| Test Accuracy     | **96.45%**               |
| Framework         | TensorFlow / Keras       |

The model demonstrates strong classification performance and generalization capability on unseen food images.

---

## 📁 Project Structure

```text
Food-Spoilage-Classification/
│
├── Food_Spoilage_Classification.ipynb
├── food_spoiled_classifier_fine_tuned.h5
├── README.md
├── requirements.txt
│
├── results/
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
└── dataset/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Food-Spoilage-Classification.git
cd Food-Spoilage-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Load the trained model:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "food_spoiled_classifier_fine_tuned.h5"
)
```

Make predictions:

```python
predictions = model.predict(image)
```

---

## 🎯 Applications

* Food Quality Inspection
* Smart Agriculture
* Warehouse Monitoring
* Retail Inventory Management
* Automated Food Sorting Systems
* Supply Chain Quality Control

---

## 🔮 Future Enhancements

* Real-time webcam-based detection
* Mobile application deployment
* Cloud-based prediction service
* IoT-enabled food monitoring systems
* Explainable AI visualization techniques

---

## 👨‍💻 Author

Durki Poshanna

---

## ⭐ Acknowledgements

This project utilizes TensorFlow, Keras, and the MobileNetV2 architecture for efficient image classification. Special thanks to the open-source community for providing valuable resources and tools for deep learning research and development.
