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
from bson.objectid import ObjectId  # ADD THIS IMPORT
import json  # ADD THIS IMPORT

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# MongoDB connection with better error handling
def get_mongodb_connection():
    max_retries = 10
    retry_delay = 5
    
    for i in range(max_retries):
        try:
            client = MongoClient(
                os.getenv('MONGO_URI', 'mongodb://mongodb:27017/fraud_detection'),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test the connection
            client.admin.command('ismaster')
            logging.info("✅ Connected to MongoDB successfully")
            return client.fraud_detection
        except Exception as e:
            logging.warning(f"Attempt {i+1}/{max_retries}: MongoDB connection failed - {e}")
            if i < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("❌ Could not connect to MongoDB after all retries")
                return None

# MinIO connection with better error handling  
def get_minio_client():
    max_retries = 10
    retry_delay = 5
    
    for i in range(max_retries):
        try:
            minio_client = Minio(
                os.getenv('MINIO_URL', 'minio:9000').replace('http://', ''),
                access_key=os.getenv('MINIO_ACCESS_KEY', 'minio'),
                secret_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),
                secure=False
            )
            # Test the connection by listing buckets
            minio_client.list_buckets()
            logging.info("✅ Connected to MinIO successfully")
            return minio_client
        except Exception as e:
            logging.warning(f"Attempt {i+1}/{max_retries}: MinIO connection failed - {e}")
            if i < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("❌ Could not connect to MinIO after all retries")
                return None

# Initialize connections
db = get_mongodb_connection()
minio_client = get_minio_client()

@app.route('/api/data/transactions', methods=['POST'])
def store_transaction():
    try:
        if db is None:
            return jsonify({"error": "Database not available"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        result = db.transactions.insert_one(data)
        return jsonify({
            "status": "success", 
            "id": str(result.inserted_id),
            "message": "Transaction stored successfully"
        })
    except Exception as e:
        logging.error(f"Error storing transaction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/transactions/<transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    try:
        if db is None:
            return jsonify({"error": "Database not available"}), 500
            
        transaction = db.transactions.find_one({"_id": ObjectId(transaction_id)})
        if transaction:
            transaction['_id'] = str(transaction['_id'])
            return jsonify(transaction)
        return jsonify({"error": "Transaction not found"}), 404
    except Exception as e:
        logging.error(f"Error retrieving transaction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/upload', methods=['POST'])
def upload_dataset():
    try:
        if minio_client is None:
            return jsonify({"error": "MinIO not available"}), 500
            
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Read the file
        df = pd.read_csv(file)
        
        # Store in MinIO
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)
        
        file_name = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        minio_client.put_object('fraud-data', file_name, csv_buffer, len(csv_bytes))
        
        return jsonify({
            "status": "success", 
            "file_name": file_name, 
            "rows": len(df),
            "columns": list(df.columns)
        })
    except Exception as e:
        logging.error(f"Error uploading dataset: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/health', methods=['GET'])
def health():
    db_status = "connected" if db is not None else "disconnected"
    minio_status = "connected" if minio_client is not None else "disconnected"
    
    return jsonify({
        "status": "healthy",
        "service": "data-service",
        "database": db_status,
        "minio": minio_status,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "FraudShield Data Service",
        "endpoints": {
            "health": "/api/data/health",
            "store_transaction": "/api/data/transactions (POST)",
            "get_transaction": "/api/data/transactions/<id> (GET)",
            "upload_data": "/api/data/upload (POST)"
        }
    })

if __name__ == '__main__':
    # Wait a bit for dependencies to be ready
    time.sleep(10)
    app.run(host='0.0.0.0', port=5000, debug=False)