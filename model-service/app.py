# Dissertation/model-service/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging
from minio import Minio
from io import BytesIO
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

model = None
scaler = None

def get_minio_client():
    return Minio(
        os.getenv('MINIO_URL', 'minio:9000').replace('http://', ''),
        access_key=os.getenv('MINIO_ACCESS_KEY', 'minio'),
        secret_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),
        secure=False
    )

def load_model():
    global model, scaler
    try:
        minio_client = get_minio_client()
        
        # Try to load existing model
        try:
            model_response = minio_client.get_object('fraud-models', 'best_fraud_model.pkl')
            model = joblib.load(BytesIO(model_response.read()))
            model_response.close()
            
            scaler_response = minio_client.get_object('fraud-models', 'scaler.pkl')
            scaler = joblib.load(BytesIO(scaler_response.read()))
            scaler_response.close()
            
            logging.info("✅ Model loaded from MinIO")
        except:
            # Create a simple model for demo
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Create dummy data for initial model
            X_dummy = np.random.rand(100, 10)
            y_dummy = np.random.randint(0, 2, 100)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dummy)
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_scaled, y_dummy)
            
            logging.info("✅ Created demo model")
            
    except Exception as e:
        logging.error(f"Error loading model: {e}")

@app.route('/api/model/predict', methods=['POST'])
def predict():
    global model, scaler
    
    if model is None:
        load_model()
    
    try:
        features = request.json['features']
        feature_vector = np.array([list(features.values())])
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'is_fraud': bool(prediction == 1)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)