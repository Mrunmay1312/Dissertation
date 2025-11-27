# data-service/app.py
from flask import Flask, request, jsonify
import pandas as pd
from pymongo import MongoClient
import joblib
import os

app = Flask(__name__)

# MongoDB connection
client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
db = client.fraud_detection

@app.route('/api/data/transactions', methods=['POST'])
def store_transaction():
    data = request.json
    db.transactions.insert_one(data)
    return jsonify({"status": "success", "id": str(data['_id'])})

@app.route('/api/data/transactions/<transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    transaction = db.transactions.find_one({"_id": transaction_id})
    return jsonify(transaction)

@app.route('/api/data/batch', methods=['POST'])
def upload_batch_data():
    file = request.files['file']
    df = pd.read_csv(file)
    records = df.to_dict('records')
    db.transactions.insert_many(records)
    return jsonify({"status": "success", "count": len(records)})
