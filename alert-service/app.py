# Dissertation/alert-service/app.py
from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MimeText
import os
import logging
import json
from datetime import datetime
from minio import Minio
from io import BytesIO

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def get_minio_client():
    return Minio(
        os.getenv('MINIO_URL', 'minio:9000').replace('http://', ''),
        access_key=os.getenv('MINIO_ACCESS_KEY', 'minio'),
        secret_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),
        secure=False
    )

minio_client = get_minio_client()

@app.route('/api/alerts/fraud', methods=['POST'])
def send_fraud_alert():
    try:
        alert_data = request.json
        
        # Log alert to MinIO
        alert_log = {
            "transaction_id": alert_data.get('transaction_id'),
            "probability": alert_data.get('probability'),
            "amount": alert_data.get('amount'),
            "timestamp": datetime.now().isoformat(),
            "alert_sent": True
        }
        
        log_data = json.dumps(alert_log).encode('utf-8')
        log_buffer = BytesIO(log_data)
        
        file_name = f"alerts/{datetime.now().strftime('%Y%m%d')}/{alert_data.get('transaction_id')}.json"
        minio_client.put_object('logs', file_name, log_buffer, len(log_data))
        
        # Send email if configured
        if alert_data.get('send_email', False):
            send_email_alert(alert_data)
        
        return jsonify({
            "status": "success",
            "alert_logged": True,
            "alert_id": alert_data.get('transaction_id')
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def send_email_alert(alert_data):
    # This is a stub - implement actual email sending
    logging.info(f"Would send email alert for: {alert_data.get('transaction_id')}")

@app.route('/api/alerts/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "alert-service"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)