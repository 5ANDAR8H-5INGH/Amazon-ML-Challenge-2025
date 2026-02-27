from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import joblib
import io
import re
import ftfy
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

app = FastAPI()

# Allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
model = joblib.load("model1.pkl")
vectorizer = joblib.load("vectorizer.pkl")
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
EMBSIZE = 2048
print("Models loaded successfully!")

def preprocess_text(text):
    text = ftfy.fix_text(str(text))
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

def extract_features(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet.predict(x, verbose=0)
    return features.flatten()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Product Price Predictor</title>
    </head>
    <body style="font-family: Arial; padding: 40px;">
        <h2>🛒 Product Price Predictor</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label>Upload Product Image:</label><br>
            <input type="file" name="image" required><br><br>

            <label>Quantity:</label><br>
            <input type="number" name="quantity" value="1" min="1" required><br><br>

            <label>Description (optional):</label><br>
            <input type="text" name="description"><br><br>

            <button type="submit">Predict Price</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    image: UploadFile = File(...),
    quantity: int = Form(...),
    description: str = Form("")
):
    image_bytes = await image.read()

    text = preprocess_text(description)
    text_feat = vectorizer.transform([text])
    img_feat = extract_features(image_bytes)

    X = hstack([text_feat, csr_matrix(img_feat.reshape(1, -1))])
    log_price = model.predict(X)[0]

    price_per = np.expm1(log_price)
    price_per = np.clip(price_per, 1.0, None)
    total = price_per * quantity

    return f"""
    <h2>💰 Prediction Result</h2>
    <p><b>Price per unit:</b> ${price_per:.2f}</p>
    <p><b>Total price:</b> ${total:.2f}</p>
    <br><a href="/">Go Back</a>
    """

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
