import os
import numpy as np
import joblib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from flask_ngrok import run_with_ngrok

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize to consistent size
            features = extract_features(img)  # define this below
            images.append(features)
            labels.append(label)
    return images, labels

# Example color slot feature extractor (HSV mean per stripe)
def extract_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])
    v_mean = np.mean(hsv[:,:,2])
    return [h_mean, s_mean, v_mean]

correct_X, correct_y = load_images_from_folder("/content/dataset/correct", 1)
incorrect_X, incorrect_y = load_images_from_folder("/content/dataset/Incorrect", 0)

X = correct_X + incorrect_X
y = incorrect_y + correct_y

X = np.array(X)
y = np.array(y)


IMG_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'
CORRECT_DIR = "dataset/correct"
INCORRECT_DIR = "dataset/Incorrect"
AUG_PER_IMAGE = 100

app = Flask(__name__)
run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('saved_model', exist_ok=True)

# Model paths
MODEL_PATH = 'saved_model/model.pkl'
FEATURE_NET_PATH = 'saved_model/feature_net.h5'

# Load or train model
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NET_PATH):
    clf = joblib.load(MODEL_PATH)
    feature_net = load_model(FEATURE_NET_PATH)
    print("✅ Model and FeatureNet loaded from disk.")
else:
    print("⏳ Training model, please wait...")

    # Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.9, 1.1),
        shear_range=0.1,
        vertical_flip=False,
        preprocessing_function=preprocess_input
    )

    def augment_images(folder_path, label):
        features, labels = [], []
        for file in tqdm(os.listdir(folder_path), desc=f"Augmenting {label}"):
            img_path = os.path.join(folder_path, file)
            img = load_img(img_path, target_size=IMG_SIZE)
            x = img_to_array(img).reshape((1,) + img_to_array(img).shape)

            count = 0
            for batch in datagen.flow(x, batch_size=1):
                features.append(batch[0])
                labels.append(label)
                count += 1
                if count >= AUG_PER_IMAGE:
                    break
        return np.array(features), np.array(labels)

    correct_imgs, correct_labels = augment_images(CORRECT_DIR, label=1)
    incorrect_imgs, incorrect_labels = augment_images(INCORRECT_DIR, label=0)

    X_images = np.concatenate([correct_imgs, incorrect_imgs], axis=0)
    y_labels = np.concatenate([correct_labels, incorrect_labels], axis=0)

    # Feature extractor
    feature_net = EfficientNetB0(include_top=False, pooling='avg',
                                 input_shape=(224, 224, 3), weights='imagenet')

    X_features = feature_net.predict(X_images, batch_size=32, verbose=1)

    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save model
    # joblib.dump(clf, MODEL_PATH)
    # feature_net.save(FEATURE_NET_PATH)
    print("✅ Model trained and saved.")

# Prediction function
def predict_cable_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = preprocess_input(img_to_array(img))[None, ...]
    feat = feature_net.predict(arr)
    prob = clf.predict_proba(feat)[0][1]
    label = "Correct" if prob >= 0.5 else "Incorrect"
    return label, float(prob)

# Main route
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

if __name__ == '__main__':
    app.run()
