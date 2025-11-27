# feature-service/app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from geopy.distance import geodesic
import yaml
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load configuration
with open('/app/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class FeatureEngine:
    def __init__(self, config):
        self.config = config
        self.numerical_features = config['features']['numerical']
        self.categorical_features = config['features']['categorical']
        self.derived_features = config['features']['derived']
        
    def extract_features(self, transaction):
        features = {}
        
        # Basic numerical features
        for feature in self.numerical_features:
            if feature in transaction:
                features[feature] = float(transaction[feature])
        
        # Categorical features
        for feature in self.categorical_features:
            if feature in transaction:
                features[feature] = int(transaction[feature])
        
        # Derived features
        self._add_derived_features(features, transaction)
        
        # Risk indicators
        self._add_risk_indicators(features, transaction)
        
        return features
    
    def _add_derived_features(self, features, transaction):
        # Amount-based features
        if 'amount' in transaction:
            features['amount_log'] = np.log1p(float(transaction['amount']))
            features['amount_squared'] = float(transaction['amount']) ** 2
        
        # Time-based features
        if 'timestamp' in transaction:
            try:
                transaction_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
                features['hour'] = transaction_time.hour
                features['day_of_week'] = transaction_time.weekday()
                features['is_weekend'] = 1 if transaction_time.weekday() >= 5 else 0
                features['hour_sin'] = np.sin(2 * np.pi * transaction_time.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * transaction_time.hour / 24)
            except Exception as e:
                logging.warning(f"Error parsing timestamp: {e}")
        
        # Location-based features
        if all(k in transaction for k in ['user_home_location', 'transaction_location']):
            try:
                home_loc = (transaction['user_home_location']['lat'], 
                           transaction['user_home_location']['lon'])
                tx_loc = (transaction['transaction_location']['lat'], 
                         transaction['transaction_location']['lon'])
                features['distance_from_home'] = geodesic(home_loc, tx_loc).km
            except Exception as e:
                logging.warning(f"Error calculating distance: {e}")
                features['distance_from_home'] = 0.0
        
        # Behavioral features
        if 'user_tx_count_24h' in transaction:
            features['tx_frequency_24h'] = float(transaction['user_tx_count_24h'])
        
        if 'user_avg_amount_7d' in transaction:
            current_amount = float(transaction.get('amount', 0))
            avg_amount = float(transaction['user_avg_amount_7d'])
            if avg_amount > 0:
                features['amount_ratio_to_avg'] = current_amount / avg_amount
    
    def _add_risk_indicators(self, features, transaction):
        # High amount risk
        if 'amount' in transaction:
            amount = float(transaction['amount'])
            high_risk_threshold = self.config['thresholds']['high_risk_amount']
            features['is_high_amount'] = 1 if amount > high_risk_threshold else 0
        
        # Suspicious time risk
        if 'hour' in features:
            suspicious_hours = self.config['thresholds']['suspicious_hours']
            features['is_suspicious_hour'] = 1 if features['hour'] in suspicious_hours else 0
        
        # High frequency risk
        if 'tx_frequency_24h' in features:
            features['is_high_frequency'] = 1 if features['tx_frequency_24h'] > 10 else 0

feature_engine = FeatureEngine(config)

@app.route('/api/features/extract', methods=['POST'])
def extract_features_endpoint():
    try:
        transaction = request.json
        if not transaction:
            return jsonify({"error": "No transaction data provided"}), 400
        
        features = feature_engine.extract_features(transaction)
        
        logging.info(f"Extracted {len(features)} features for transaction")
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
        transactions = request.json.get('transactions', [])
        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400
        
        features_list = []
        for transaction in transactions:
            features = feature_engine.extract_features(transaction)
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

@app.route('/metrics', methods=['GET'])
def metrics():
    # Basic metrics endpoint for Prometheus
    from prometheus_client import generate_latest, Counter, Histogram
    import prometheus_client
    
    # Define metrics
    FEATURE_EXTRACTION_COUNT = Counter(
        'feature_extraction_requests_total',
        'Total number of feature extraction requests'
    )
    
    FEATURE_EXTRACTION_DURATION = Histogram(
        'feature_extraction_duration_seconds',
        'Time spent processing feature extraction'
    )
    
    return generate_latest(prometheus_client.REGISTRY)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
