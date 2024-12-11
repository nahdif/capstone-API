from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
import numpy as np
import requests
import os

# Flask app
app = Flask(__name__)

# URLs untuk model penyakit jantung
HEART_DISEASE_MODEL_JSON_URL = "https://storage.googleapis.com/api-data-capstone/model_architecture.json"
HEART_DISEASE_MODEL_WEIGHTS_URL = "https://storage.googleapis.com/api-data-capstone/model.weights.h5"

# Path lokal untuk menyimpan model
MODEL_JSON_PATH = "heart_disease_model.json"
MODEL_WEIGHTS_PATH = "heart_disease_model_weights.h5"

# Fungsi untuk mengunduh file
def download_file(url, destination):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError untuk kode status buruk
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded: {destination}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file from {url}: {e}")

# Unduh model JSON dan weights jika belum ada
if not os.path.exists(MODEL_JSON_PATH):
    download_file(HEART_DISEASE_MODEL_JSON_URL, MODEL_JSON_PATH)

if not os.path.exists(MODEL_WEIGHTS_PATH):
    download_file(HEART_DISEASE_MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH)

# Muat model dari JSON dan weights
model = None
try:
    with open(MODEL_JSON_PATH, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Endpoint untuk prediksi penyakit jantung
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Pastikan input berupa array dengan panjang sesuai (misalnya 22 fitur untuk model penyakit jantung)
        input_data = data.get('features')
        if not input_data or len(input_data) != 22:  # Ganti 22 dengan jumlah fitur yang benar
            return jsonify({"error": "Invalid input, 'features' must be an array of 22 numeric values"}), 400

        # Validasi setiap elemen adalah numerik
        try:
            input_array = np.array(input_data, dtype=float).reshape(1, -1)
        except ValueError:
            return jsonify({"error": "All 'features' elements must be numeric"}), 400

        # Lakukan prediksi
        prediction = model.predict(input_array)
        result = float(prediction[0][0])  # Konversi ke float

        # Tentukan kategori berdasarkan threshold 0.5
        category = "High Risk" if result >= 0.5 else "Low Risk"

        # Kembalikan hasil
        return jsonify({"prediction": result, "category": category})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk memeriksa kesehatan API
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

# Main driver
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
