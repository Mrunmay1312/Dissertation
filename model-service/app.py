# model-service/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

app = Flask(__name__)

# Global model variable
model = None
SCALER = None

def load_model():
    global model, SCALER
    try:
        model = joblib.load('/app/models/fraud_model.pkl')
        SCALER = joblib.load('/app/models/scaler.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/api/model/predict', methods=['POST'])
def predict():
    if model is None:
        load_model()
    
    features = request.json['features']
    feature_vector = np.array([list(features.values())])
    
    # Scale features
    feature_vector_scaled = SCALER.transform(feature_vector)
    
    # Make prediction
    prediction = model.predict(feature_vector_scaled)[0]
    probability = model.predict_proba(feature_vector_scaled)[0][1]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability),
        'is_fraud': bool(prediction == 1)
    })

@app.route('/api/model/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})
