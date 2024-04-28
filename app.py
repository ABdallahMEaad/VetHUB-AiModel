from fastapi import FastAPI, UploadFile, File
from typing import List
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Load the model
def custom_loss_deserializer(config):
    return tf.keras.losses.deserialize({'class_name': 'SparseCategoricalCrossentropy', 'config': config})

model = tf.keras.models.load_model('./4s.h5', custom_objects={'SparseCategoricalCrossentropy': custom_loss_deserializer})

# Define class names
class_names = ["Foot and Mouth Disease", "Infectious Bovine Keratoconjunctivitis", "Lumpy Skin", "Normal"]

# Define class-specific information
class_info = {
    "Foot and Mouth Disease": {
        "Cause": "Caused by a virus.",
        "Symptoms": "Fever, blisters on the tongue and lips, drooling, lameness.",
        "Treatment": "No specific treatment, supportive care, vaccination for prevention."
    },
    "Infectious Bovine Keratoconjunctivitis": {
        "Cause": "Bacterial infection.",
        "Symptoms": "Watery eyes, squinting, corneal ulceration.",
        "Treatment": "Antibiotics, eye ointments, fly control."
    },
    "Lumpy Skin": {
        "Cause": "Caused by a virus.",
        "Symptoms": "Firm nodules on the skin, fever, loss of appetite.",
        "Treatment": "Supportive care, vaccination for prevention."
    },
    "Normal": {
        "Info": "The image appears to be normal, without any detected disease."
    }
}

# Define FastAPI app
app = FastAPI()

# Prediction function
def predict(image: Image.Image):
    img = image.resize((256, 256))  # Resize image to match model input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# Define FastAPI route
@app.post("/predict/", response_model=List[dict])
async def predict_endpoint(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            predicted_class, confidence = predict(image)
            result = {"filename": file.filename, "predicted_class": predicted_class, "confidence": confidence}
            if predicted_class in class_info:
                result["class_info"] = class_info[predicted_class]
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return results
