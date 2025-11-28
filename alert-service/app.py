# Dissertation/alert-service/app.py
from flask import Flask, request, jsonify
import logging
import json
from datetime import datetime
from io import BytesIO
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Simple alert service without external dependencies for now
class AlertService:
    def __init__(self):
        self.alerts_sent = 0
    
    def send_alert(self, alert_data):
        """Send fraud alert (simulated)"""
        try:
            # Log the alert
            alert_log = {
                "transaction_id": alert_data.get('transaction_id', 'unknown'),
                "probability": alert_data.get('probability', 0),
                "amount": alert_data.get('amount', 0),
                "user_id": alert_data.get('user_id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "alert_type": "fraud_detected"
            }
            
            # Print alert to logs (in production, this would send email/SMS/etc.)
            logging.info(f"🚨 FRAUD ALERT: {alert_log}")
            
            self.alerts_sent += 1
            
            return {
                "status": "success",
                "alert_id": alert_data.get('transaction_id'),
                "message": "Alert processed successfully",
                "alert_number": self.alerts_sent
            }
            
        except Exception as e:
            logging.error(f"Alert sending error: {e}")
            return {"status": "error", "message": str(e)}

alert_service = AlertService()

@app.route('/api/alerts/fraud', methods=['POST'])
def send_fraud_alert():
    try:
        alert_data = request.json
        if not alert_data:
            return jsonify({"error": "No alert data provided"}), 400
        
        result = alert_service.send_alert(alert_data)
        
        return jsonify({
            "status": "success",
            "result": result
        })
    
    except Exception as e:
        logging.error(f"Alert endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "alert-service",
        "alerts_sent": alert_service.alerts_sent,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/alerts/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_alerts_sent": alert_service.alerts_sent,
        "service_uptime": "running",
        "last_alert_time": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "FraudShield Alert Service",
        "endpoints": {
            "health": "/api/alerts/health",
            "send_alert": "/api/alerts/fraud (POST)",
            "stats": "/api/alerts/stats (GET)"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)