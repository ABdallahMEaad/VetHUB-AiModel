# 🐄 Cattle Disease Detection using Deep Learning & FastAPI

## 📌 Project Overview

This project is an AI-powered cattle disease detection system that uses Deep Learning and Computer Vision to identify common cattle diseases from images.

The model is trained using TensorFlow/Keras and deployed through a FastAPI REST API, allowing users to upload one or multiple images and receive disease predictions along with detailed disease information.

---

## 🚀 Features

* Upload one or multiple cattle images.
* Deep Learning image classification.
* FastAPI REST API deployment.
* Disease information returned with prediction.
* Confidence-based classification.
* CORS enabled for frontend integration.
* TensorFlow/Keras model inference.

---

## 🦠 Supported Classes

| Disease                                | Description                                   |
| -------------------------------------- | --------------------------------------------- |
| Foot and Mouth Disease                 | Viral disease affecting cloven-hoofed animals |
| Infectious Bovine Keratoconjunctivitis | Bacterial eye infection (Pink Eye)            |
| Lumpy Skin Disease                     | Viral disease causing skin nodules            |
| Normal                                 | Healthy cattle image                          |

---

## 🛠️ Tech Stack

### Backend

* FastAPI
* Python
* TensorFlow
* Keras
* NumPy
* Pillow (PIL)

### Machine Learning

* Convolutional Neural Networks (CNN)
* Image Augmentation
* Transfer Learning Techniques

### Deployment

* Uvicorn
* FastAPI

---

## 📂 Project Structure

```text
project/
│
├── app.py
├── 4s.h5
├── requirements.txt
├── README.md
│
└── dataset/
    ├── train/
    ├── valid/
    └── test/
```

---

## 🧠 Model Training

### Image Specifications

```python
IMAGE_SIZE = 256
CHANNELS = 3
```

### Data Augmentation

```python
ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)
```

### Dataset Loading

```python
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(256,256),
    batch_size=32,
    class_mode='sparse'
)
```

---

## ⚙️ API Endpoint

### POST /predict/

Upload one or multiple images for disease detection.

### Request

```http
POST /predict/
```

### Form Data

```text
files: image1.jpg
files: image2.jpg
```

---

## Example Response

```json
[
  {
    "predicted_class": "Lumpy Skin",
    "class_info": {
      "Cause": "Caused by the poxvirus.",
      "Symptoms": "Firm nodules on the skin, fever, loss of appetite.",
      "Transmission": "Direct contact with infected animals.",
      "Prevention": "Vaccination and biosecurity measures.",
      "Treatment": "Supportive care and management."
    }
  }
]
```

---

## 🩺 Disease Information Included

### Foot and Mouth Disease

**Cause**

* Aphthovirus

**Symptoms**

* Fever
* Mouth blisters
* Excessive salivation
* Lameness

**Prevention**

* Vaccination
* Biosecurity

---

### Infectious Bovine Keratoconjunctivitis

**Cause**

* Moraxella bovis

**Symptoms**

* Watery eyes
* Corneal ulcers
* Photophobia

**Prevention**

* Fly control
* Animal isolation

---

### Lumpy Skin Disease

**Cause**

* Poxvirus

**Symptoms**

* Skin nodules
* Fever
* Appetite loss

**Prevention**

* Vaccination
* Biosecurity measures

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Cattle-Disease-Detection.git

cd Cattle-Disease-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn app:app --reload
```

API will start at:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc Documentation:

```text
http://127.0.0.1:8000/redoc
```

---

## 📊 Model Performance

| Metric     | Value      |
| ---------- | ---------- |
| Classes    | 4          |
| Image Size | 256x256    |
| Framework  | TensorFlow |
| API        | FastAPI    |

---

## 🔮 Future Improvements

* Mobile Application Integration
* Real-time Camera Detection
* More Disease Classes
* Explainable AI (Grad-CAM)
* Docker Deployment
* Cloud Hosting (AWS/Azure/GCP)

---

## 👨‍💻 Author

**Abdallah Me3ad**

QA Engineer | Network Engineer | AI & Computer Vision Enthusiast

GitHub: https://github.com/ABdallahMEaad

LinkedIn:https://www.linkedin.com/in/abdallah-meaad/?skipRedirect=true

---

## 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.
