import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Constants
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'saved_model/model.pkl'
FEATURE_NET_PATH = 'saved_model/feature_net.h5'

# Flask App Setup
app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Models
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NET_PATH):
    clf = joblib.load(MODEL_PATH)
    feature_net = load_model(FEATURE_NET_PATH)
    print("✅ Model and FeatureNet loaded.")
else:
    print("❌ Missing model files.")
    exit()

# Prediction function
def predict_cable_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = preprocess_input(img_to_array(img))[None, ...]
    feat = feature_net.predict(arr)
    prob = clf.predict_proba(feat)[0][1]
    label = "Correct" if prob >= 0.5 else "Incorrect"
    return label, float(prob)

# Homepage
@app.route("/", methods=["GET"])
def home():
    return "<h2>✅ Cable Sequence Detection Backend is Running</h2>"

# API Route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence = predict_cable_image(filepath)
    return jsonify({
        "result": label,
        "confidence": round(confidence, 2)
    })

# Run locally or on Render
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=10000)
