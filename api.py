import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import os


APP = FastAPI(title="Skin Disease Classifier API")

# Load model with error handling for deployment
try:
    MODEL = tf.keras.models.load_model("best_model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL = None

IMG_SIZE = (224, 224)

# Get class names - fallback for deployment
try:
    CLASS_NAMES = sorted([d for d in os.listdir("dataset/train") if os.path.isdir(os.path.join("dataset/train", d))])
except:
    CLASS_NAMES = ["acne", "eczema", "healthy", "psoriasis", "ringworm"]


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@APP.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model not loaded. Please check server logs."}
    
    try:
        content = await file.read()
        x = preprocess(content)
        preds = MODEL.predict(x, verbose=0)
        probs = preds[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probs[predicted_class_idx])
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": {
                class_name: float(prob) 
                for class_name, prob in zip(CLASS_NAMES, probs)
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=8000)


