from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRUIT_MODEL_PATH = r"C:\calAI\backend\fruit_classifier_best.pth"
FOOD_MODEL_PATH = r"C:\calAI\backend\food_classifier_best.pth"

CALORIES = {
    "Apple": 52,
    "Banana": 96,
    "Orange": 47,
    "Samosa": 262,
    "Pizza": 266,
    "Burger": 295
}

# =========================
# APP INIT
# =========================
STATIC_DIR = r"C:\calAI\static"  # <-- path to your index.html
app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL FUNCTION
# =========================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, classes

# Load models
fruit_model, fruit_classes = load_model(FRUIT_MODEL_PATH)
food_model, food_classes = load_model(FOOD_MODEL_PATH)

print("✅ Fruit and Food models loaded successfully")

# =========================
# IMAGE TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# PREDICTION FUNCTION
# =========================
def predict(model, classes, tensor):
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        top_p, top_i = torch.topk(probs, TOP_K)

    results = []
    for p, i in zip(top_p[0], top_i[0]):
        label = classes[i]
        results.append({
            "class": label,
            "confidence": round(p.item() * 100, 2),
            "calories": CALORIES.get(label, "N/A")
        })
    return results

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img = Image.open(request.files["file"]).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Predict both models
        fruit_preds = predict(fruit_model, fruit_classes, tensor)
        food_preds = predict(food_model, food_classes, tensor)

        # Decide type based on top confidence
        top_fruit_conf = fruit_preds[0]["confidence"]
        top_food_conf = food_preds[0]["confidence"]

        if top_fruit_conf >= top_food_conf:
            final_preds = fruit_preds
            kind = "Fruit"
        else:
            final_preds = food_preds
            kind = "Food"

        return jsonify({
            "type": kind,
            "top_prediction": final_preds[0],
            "fruit_confidence": top_fruit_conf,
            "food_confidence": top_food_conf,
            "top_5": final_preds,
            "fruit_top_5": fruit_preds,
            "food_top_5": food_preds
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
