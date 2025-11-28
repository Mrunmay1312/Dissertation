# Dissertation/feature-service/app.py
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def extract_features(transaction):
    """Extract features from transaction data"""
    features = {}
    
    try:
        # Transaction amount features
        amount = float(transaction.get('amount', 0))
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount) if amount > 0 else 0
        
        # Time-based features
        timestamp = transaction.get('timestamp', datetime.now().isoformat())
        try:
            transaction_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            features['hour'] = transaction_time.hour
            features['day_of_week'] = transaction_time.weekday()
            features['is_weekend'] = 1 if transaction_time.weekday() >= 5 else 0
        except:
            # Use current time if parsing fails
            now = datetime.now()
            features['hour'] = now.hour
            features['day_of_week'] = now.weekday()
            features['is_weekend'] = 1 if now.weekday() >= 5 else 0
        
        # User behavior features
        features['transaction_count_24h'] = float(transaction.get('user_tx_count_24h', 0))
        features['avg_amount_7d'] = float(transaction.get('user_avg_amount_7d', 0))
        
        # Location features
        features['distance_from_home'] = float(transaction.get('distance_from_home', 0))
        
        # Risk indicators
        features['is_high_amount'] = 1 if amount > 1000 else 0
        features['is_suspicious_hour'] = 1 if features['hour'] in [0, 1, 2, 3, 4, 5] else 0
        features['is_high_frequency'] = 1 if features['transaction_count_24h'] > 10 else 0
        
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        # Return default features if extraction fails
        features = {
            'amount': 0, 'amount_log': 0, 'hour': 12, 'day_of_week': 0, 
            'is_weekend': 0, 'transaction_count_24h': 0, 'avg_amount_7d': 0,
            'distance_from_home': 0, 'is_high_amount': 0, 'is_suspicious_hour': 0,
            'is_high_frequency': 0
        }
    
    return features

@app.route('/api/features/extract', methods=['POST'])
def extract_features_endpoint():
    try:
        transaction = request.json
        if not transaction:
            return jsonify({"error": "No transaction data provided"}), 400
        
        features = extract_features(transaction)
        
        return jsonify({
            "features": features,
            "feature_count": len(features),
            "status": "success"
        })
    
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/features/batch', methods=['POST'])
def extract_batch_features():
    try:
        data = request.json
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400
        
        features_list = []
        for transaction in transactions:
            features = extract_features(transaction)
            features_list.append(features)
        
        return jsonify({
            "features": features_list,
            "transaction_count": len(features_list),
            "status": "success"
        })
    
    except Exception as e:
        logging.error(f"Batch feature extraction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/features/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "feature-service",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "FraudShield Feature Service",
        "endpoints": {
            "health": "/api/features/health",
            "extract_features": "/api/features/extract (POST)",
            "batch_extract": "/api/features/batch (POST)"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)