from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# =========================
# PERFORMANCE SAFETY (FREE PLAN)
# =========================
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
TOP_K = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = BASE_DIR  # index.html is in same folder

FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "fruit_classifier_best.pth")
FOOD_MODEL_PATH = os.path.join(BASE_DIR, "food_classifier_best.pth")

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
app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(classes)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, classes

fruit_model, fruit_classes = load_model(FRUIT_MODEL_PATH)
food_model, food_classes = load_model(FOOD_MODEL_PATH)

print("✅ Models loaded successfully")

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# PREDICTION
# =========================
def predict(model, classes, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
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
        image = Image.open(request.files["file"]).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        fruit_preds = predict(fruit_model, fruit_classes, tensor)
        food_preds = predict(food_model, food_classes, tensor)

        if fruit_preds[0]["confidence"] >= food_preds[0]["confidence"]:
            final = fruit_preds
            kind = "Fruit"
        else:
            final = food_preds
            kind = "Food"

        return jsonify({
            "type": kind,
            "top_prediction": final[0],
            "top_5": final,
            "fruit_top_5": fruit_preds,
            "food_top_5": food_preds
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
