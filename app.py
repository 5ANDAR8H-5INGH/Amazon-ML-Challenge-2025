import gradio as gr
import numpy as np
import lightgbm as lgb
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import re
import ftfy

print("🔄 Loading models...")
model = lgb.Booster(model_file='model.lgb')
vectorizer = joblib.load('vectorizer.pkl')
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
EMBSIZE = 2048
print("✅ Models loaded!")

def preprocess_text(text):
    text = ftfy.fix_text(str(text))
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().replace(',', '')
    return text

def extract_features(image):
    if image is None:
        return np.zeros(EMBSIZE)
    try:
        img = load_img(image, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)
        return resnet.predict(x, verbose=0).flatten()
    except:
        return np.zeros(EMBSIZE)

def predict_price(image, quantity, description=""):
    text = preprocess_text(description)
    text_feat = vectorizer.transform([text])
    img_emb = extract_features(image)
    X = hstack([text_feat, csr_matrix(img_emb.reshape(1, -1))])
    log_price = model.predict(X)[0]
    price_per = np.expm1(log_price)
    price_per = np.clip(price_per, 1.0, None)
    total = price_per * quantity
    return f"**Predicted Total: ${total:.2f}**\n(per unit: ${price_per:.2f})"

with gr.Blocks(title="Product Price Predictor") as demo:
    gr.Markdown("# 🛒 Product Price Predictor")
    gr.Markdown("Upload image + pack quantity → Get price prediction")
    
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="📸 Product Image")
            quantity = gr.Number(label="📦 Pack Quantity", value=1, minimum=1)
            desc = gr.Textbox(label="📝 Description (optional)", 
                            placeholder="e.g., 'basil 6.25 oz herb spice'")
            predict_btn = gr.Button("🔮 Predict Price", variant="primary")
        
        output = gr.Textbox(label="💰 Prediction", lines=2)
    
    predict_btn.click(predict_price, inputs=[image, quantity, desc], outputs=output)

if __name__ == "__main__":
    demo.launch()
