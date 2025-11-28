# Dissertation/feature-service/app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import requests

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://data-service:5000')

def extract_features(transaction):
    features = {}
    
    # Transaction amount features
    features['amount'] = transaction.get('amount', 0)
    features['amount_log'] = np.log1p(features['amount'])
    
    # Time-based features
    if 'timestamp' in transaction:
        try:
            transaction_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
            features['hour'] = transaction_time.hour
            features['day_of_week'] = transaction_time.weekday()
            features['is_weekend'] = 1 if transaction_time.weekday() >= 5 else 0
        except:
            features['hour'] = 12
            features['day_of_week'] = 0
            features['is_weekend'] = 0
    
    # User behavior features
    features['transaction_count_24h'] = transaction.get('user_tx_count_24h', 0)
    features['avg_amount_7d'] = transaction.get('user_avg_amount_7d', 0)
    
    # Location features (simplified)
    features['distance_from_home'] = transaction.get('distance_from_home', 0)
    
    # Risk indicators
    features['is_high_amount'] = 1 if features['amount'] > 1000 else 0
    features['is_suspicious_hour'] = 1 if features['hour'] in [0, 1, 2, 3, 4, 5] else 0
    
    return features

@app.route('/api/features/extract', methods=['POST'])
def extract_features_endpoint():
    try:
        transaction = request.json
        features = extract_features(transaction)
        return jsonify({
            "features": features,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "feature-service"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)