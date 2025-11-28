# Dissertation/data-service/app.py
from flask import Flask, request, jsonify
import pandas as pd
from pymongo import MongoClient
import joblib
import os
import logging
from minio import Minio
from io import BytesIO
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# MongoDB connection
def get_mongodb_connection():
    max_retries = 5
    for i in range(max_retries):
        try:
            client = MongoClient(
                os.getenv('MONGO_URI', 'mongodb://mongodb:27017/fraud_detection'),
                serverSelectionTimeoutMS=5000
            )
            client.admin.command('ismaster')
            logging.info("✅ Connected to MongoDB")
            return client.fraud_detection
        except Exception as e:
            logging.warning(f"Attempt {i+1}/{max_retries}: MongoDB connection failed - {e}")
            if i < max_retries - 1:
                time.sleep(2)
    raise Exception("Could not connect to MongoDB")

# MinIO connection
def get_minio_client():
    return Minio(
        os.getenv('MINIO_URL', 'minio:9000').replace('http://', ''),
        access_key=os.getenv('MINIO_ACCESS_KEY', 'minio'),
        secret_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),
        secure=False
    )

# Initialize connections
db = get_mongodb_connection()
minio_client = get_minio_client()

@app.route('/api/data/transactions', methods=['POST'])
def store_transaction():
    try:
        data = request.json
        result = db.transactions.insert_one(data)
        return jsonify({"status": "success", "id": str(result.inserted_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/transactions/<transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    try:
        from bson.objectid import ObjectId
        transaction = db.transactions.find_one({"_id": ObjectId(transaction_id)})
        if transaction:
            transaction['_id'] = str(transaction['_id'])
            return jsonify(transaction)
        return jsonify({"error": "Transaction not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/upload', methods=['POST'])
def upload_dataset():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Store in MinIO
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)
        
        file_name = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        minio_client.put_object('fraud-data', file_name, csv_buffer, len(csv_bytes))
        
        return jsonify({"status": "success", "file_name": file_name, "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "data-service",
        "timestamp": pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)