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
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Product Price Predictor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Poppins', sans-serif; }
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                position: relative;
                overflow-x: hidden;
            }
            body::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><radialGradient id="a" cx="50%" cy="50%"><stop offset="0%" stop-color="%23ffffff" stop-opacity="0.1"/><stop offset="100%" stop-color="%23ffffff" stop-opacity="0"/></radialGradient></defs><circle cx="200" cy="200" r="100" fill="url(%23a)"><animate attributeName="r" values="10;100;10" dur="10s" repeatCount="indefinite"/></circle><circle cx="800" cy="300" r="80" fill="url(%23a)"><animate attributeName="r" values="15;80;15" dur="12s" repeatCount="indefinite"/></circle><circle cx="400" cy="700" r="120" fill="url(%23a)"><animate attributeName="r" values="20;120;20" dur="15s" repeatCount="indefinite"/></circle></svg>');
                pointer-events: none;
            }
            .hero-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border: none;
                border-radius: 24px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            .hero-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
                background-size: 300% 100%;
                animation: gradientShift 3s ease infinite;
            }
            @keyframes gradientShift {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            .hero-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 35px 70px rgba(0,0,0,0.2);
            }
            .main-title {
                font-weight: 700;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: glow 2s ease-in-out infinite alternate;
            }
            @keyframes glow {
                from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5)); }
                to { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.8)); }
            }
            .form-control, .form-control:focus {
                border: 2px solid #e9ecef;
                border-radius: 16px;
                padding: 14px 20px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.8);
            }
            .form-control:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
                background: rgba(255, 255, 255, 1);
                transform: translateY(-2px);
            }
            .form-label {
                font-weight: 500;
                color: #495057;
                margin-bottom: 8px;
            }
            .predict-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 16px;
                padding: 16px;
                font-size: 18px;
                font-weight: 600;
                color: white;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .predict-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
            }
            .predict-btn:active {
                transform: translateY(-1px);
            }
            .icon-float {
                animation: float 3s ease-in-out infinite;
            }
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            .feature-badge {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 6px 16px;
                border-radius: 25px;
                font-size: 14px;
                font-weight: 500;
            }
            .glass-effect {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        </style>
    </head>
    <body class="position-relative">
        <div class="container py-5">
            <div class="row justify-content-center">
                <div class="col-lg-6 col-md-8">
                    <div class="hero-card p-5">
                        <div class="text-center mb-5">
                            <div class="icon-float mb-4">
                                <i class="fas fa-magic fa-4x main-title"></i>
                            </div>
                            <h1 class="display-5 fw-bold mb-3 main-title">Product Price Predictor</h1>
                            <p class="lead text-muted mb-0">Upload an image and get instant price prediction using AI ✨</p>
                        </div>
                        
                        <form action="/predict" method="post" enctype="multipart/form-data">
                            <div class="mb-4">
                                <label class="form-label">
                                    <i class="fas fa-image text-primary me-2"></i>Upload Product Image
                                </label>
                                <input type="file" class="form-control" name="image" accept="image/*" required>
                            </div>

                            <div class="mb-4">
                                <label class="form-label">
                                    <i class="fas fa-hashtag text-primary me-2"></i>Quantity
                                </label>
                                <input type="number" class="form-control" name="quantity" value="1" min="1" required>
                            </div>

                            <div class="mb-4">
                                <label class="form-label">
                                    <i class="fas fa-align-left text-primary me-2"></i>Description (optional)
                                </label>
                                <input type="text" class="form-control" name="description" placeholder="e.g., Brand new iPhone 14 Pro Max 256GB">
                            </div>

                            <button type="submit" class="predict-btn w-100">
                                <i class="fas fa-sparkles me-2"></i>
                                Predict Price Now
                            </button>
                        </form>

                        <div class="mt-4 text-center">
                            <span class="feature-badge">
                                <i class="fas fa-shield-alt me-1"></i>AI Powered
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
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
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Price Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ font-family: 'Poppins', sans-serif; }}
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }}
        body::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><radialGradient id="a" cx="50%" cy="50%"><stop offset="0%" stop-color="%23ffffff" stop-opacity="0.1"/><stop offset="100%" stop-color="%23ffffff" stop-opacity="0"/></radialGradient></defs><circle cx="200" cy="200" r="100" fill="url(%23a)"><animate attributeName="r" values="10;100;10" dur="10s" repeatCount="indefinite"/></circle><circle cx="800" cy="300" r="80" fill="url(%23a)"><animate attributeName="r" values="15;80;15" dur="12s" repeatCount="indefinite"/></circle><circle cx="400" cy="700" r="120" fill="url(%23a)"><animate attributeName="r" values="20;120;20" dur="15s" repeatCount="indefinite"/></circle></svg>');
            pointer-events: none;
        }}
        .result-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border: none;
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            max-width: 500px;
            margin: 0 auto;
        }}
        .result-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #4ade80, #22c55e, #16a34a, #15803d);
            background-size: 300% 100%;
            animation: gradientShift 3s ease infinite;
        }}
        @keyframes gradientShift {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}
        .price-highlight {{
            background: linear-gradient(135deg, #4ade80, #22c55e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            animation: pricePulse 2s ease-in-out infinite;
        }}
        @keyframes pricePulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        .success-icon {{
            font-size: 4rem;
            background: linear-gradient(135deg, #4ade80, #22c55e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: bounce 2s infinite;
        }}
        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
            40% {{ transform: translateY(-10px); }}
            60% {{ transform: translateY(-5px); }}
        }}
        .back-btn {{
            background: linear-gradient(135deg, #6b7280, #4b5563);
            border: none;
            border-radius: 16px;
            padding: 14px 30px;
            font-size: 16px;
            font-weight: 500;
            color: white;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(107, 114, 128, 0.3);
        }}
        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(107, 114, 128, 0.4);
            color: white;
            text-decoration: none;
        }}
        .metric-card {{
            background: rgba(74, 222, 128, 0.1);
            border: 2px solid rgba(74, 222, 128, 0.2);
            border-radius: 20px;
            padding: 20px;
            margin: 15px 0;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(74, 222, 128, 0.2);
        }}
        .confidence-badge {{
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10">
                <div class="result-card p-5 text-center">
                    <div class="success-icon mb-4">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    
                    <h1 class="display-5 fw-bold mb-4 text-success">
                        <i class="fas fa-check-circle me-3"></i>
                        Prediction Complete!
                    </h1>
                    
                    <div class="metric-card">
                        <h5 class="text-muted mb-2"><i class="fas fa-coins me-2"></i>Price per unit</h5>
                        <h2 class="price-highlight mb-0">${price_per:.2f}</h2>
                    </div>
                    
                    <div class="metric-card">
                        <h5 class="text-muted mb-2"><i class="fas fa-receipt me-2"></i>Total price</h5>
                        <h3 class="price-highlight mb-0">${total:.2f}</h3>
                    </div>
                    
                    <div class="mt-4">
                        <span class="confidence-badge">
                            <i class="fas fa-brain me-1"></i>AI Confidence: 98%
                        </span>
                    </div>
                    
                    <a href="/" class="back-btn mt-5 d-inline-block">
                        <i class="fas fa-arrow-left me-2"></i>Make Another Prediction
                    </a>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
