import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import cv2
from flask import Flask, render_template, request
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
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Pre-trained Models
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NET_PATH):
    clf = joblib.load(MODEL_PATH)
    feature_net = load_model(FEATURE_NET_PATH)
    print("✅ Model and FeatureNet loaded from disk.")
else:
    print("❌ Model files not found. Please upload 'model.pkl' and 'feature_net.h5'")
    exit()

# Prediction Function
def predict_cable_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = preprocess_input(img_to_array(img))[None, ...]
    feat = feature_net.predict(arr)
    prob = clf.predict_proba(feat)[0][1]
    label = "Correct" if prob >= 0.5 else "Incorrect"
    return label, float(prob)

# Web Interface
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, similarity = predict_cable_image(filepath)
            result = f"Prediction: {label} | Confidence: {similarity:.2f}"
            image_path = '/' + filepath

    return render_template('index.html', result=result, image_path=image_path)

# Run App
if __name__ == "__main__":
    import traceback
    try:
        app.run(debug=False, host='0.0.0.0', port=10000)
    except Exception as e:
        print("❌ CRASHED:", e)
        traceback.print_exc()
