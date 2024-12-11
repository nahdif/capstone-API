import os
import numpy as np
from flask import Flask, request, jsonify
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
import requests

app = Flask(__name__)

# URLs Model di Google Cloud Storage
MODEL_URL = "https://storage.googleapis.com/api-data-capstone/skindisease_mymodel_weights.h5"
JSON_URL = "https://storage.googleapis.com/api-data-capstone/SkinDisease_MyModel.json"

# Unduh model jika belum ada secara lokal
LOCAL_MODEL_PATH = "skin_model_weights.h5"
LOCAL_JSON_PATH = "skin_model.json"

def download_file(url, local_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for HTTP codes not 200
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {local_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

if not os.path.exists(LOCAL_JSON_PATH):
    print("Downloading model JSON...")
    download_file(JSON_URL, LOCAL_JSON_PATH)

if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading model weights...")
    download_file(MODEL_URL, LOCAL_MODEL_PATH)

# Load model
try:
    with open(LOCAL_JSON_PATH, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(LOCAL_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Labels (sesuaikan dengan kategori penyakit kulit yang ada di model)
LABELS = [ "eczema", "psoriasis", "basal cell", "Atopic Dermatitis"]

# Preprocessing function
def preprocess_image(image_path, target_size=(180, 180)):  # Sesuaikan ukuran di sini
    try:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) / 255.0  # Normalisasi
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise

# API Route: Home
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Skin Disease Prediction API!"})

# API Route: Predict
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        # Preprocess and predict
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        predicted_label = LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"prediction": predicted_label, "confidence": confidence})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
