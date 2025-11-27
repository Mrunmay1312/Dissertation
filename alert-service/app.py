# alert-service/app.py
from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os
import logging
import json
import requests
from datetime import datetime
import yaml
from jinja2 import Template
from minio import Minio

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load configuration
with open('/app/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize MinIO client
minio_client = Minio(
    "minio-service:9000",
    access_key=os.getenv('MINIO_ACCESS_KEY', 'minio'),
    secret_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),
    secure=False
)

class AlertManager:
    def __init__(self, config):
        self.config = config
        self.email_enabled = config['alerts']['email']['enabled']
        self.slack_enabled = config['alerts']['slack']['enabled']
        
    def send_alert(self, alert_data):
        """Send alerts through configured channels"""
        alerts_sent = []
        
        # Determine alert level
        probability = alert_data.get('probability', 0)
        if probability >= self.config['alerts']['thresholds']['high_risk']:
            alert_level = 'high_risk'
        elif probability >= self.config['alerts']['thresholds']['medium_risk']:
            alert_level = 'medium_risk'
        else:
            alert_level = 'low_risk'
        
        # Send email alert
        if self.email_enabled and alert_level in ['high_risk', 'medium_risk']:
            try:
                self._send_email_alert(alert_data, alert_level)
                alerts_sent.append('email')
            except Exception as e:
                logging.error(f"Failed to send email alert: {e}")
        
        # Send Slack alert
        if self.slack_enabled and alert_level == 'high_risk':
            try:
                self._send_slack_alert(alert_data)
                alerts_sent.append('slack')
            except Exception as e:
                logging.error(f"Failed to send Slack alert: {e}")
        
        # Log alert
        self._log_alert(alert_data, alert_level, alerts_sent)
        
        return {
            "alert_level": alert_level,
            "channels_used": alerts_sent,
            "timestamp": datetime.now().isoformat()
        }
    
    def _send_email_alert(self, alert_data, alert_level):
        """Send email alert"""
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        
        # Create message
        message = MimeMultipart()
        message['Subject'] = f"Fraud Alert - {alert_level.replace('_', ' ').title()}"
        message['From'] = sender_email
        message['To'] = ', '.join(self.config['alerts']['email']['recipients'])
        
        # Create HTML content
        html_template = """
        <html>
        <body>
            <h2>🚨 Fraud Detection Alert</h2>
            <p><strong>Alert Level:</strong> {{ alert_level }}</p>
            <p><strong>Transaction ID:</strong> {{ transaction_id }}</p>
            <p><strong>Amount:</strong> ${{ amount }}</p>
            <p><strong>Fraud Probability:</strong> {{ probability|round(4) }} ({{ (probability * 100)|round(2) }}%)</p>
            <p><strong>Timestamp:</strong> {{ timestamp }}</p>
            <p><strong>User ID:</strong> {{ user_id }}</p>
            <br>
            <p>Please review this transaction immediately.</p>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            alert_level=alert_level.replace('_', ' ').title(),
            transaction_id=alert_data.get('transaction_id', 'N/A'),
            amount=alert_data.get('amount', 0),
            probability=alert_data.get('probability', 0),
            timestamp=alert_data.get('timestamp', 'N/A'),
            user_id=alert_data.get('user_id', 'N/A')
        )
        
        message.attach(MimeText(html_content, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        logging.info(f"Email alert sent for transaction {alert_data.get('transaction_id')}")
    
    def _send_slack_alert(self, alert_data):
        """Send Slack alert"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            logging.warning("Slack webhook URL not configured")
            return
        
        slack_message = {
            "text": "🚨 High Risk Fraud Detected!",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 High Risk Fraud Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Transaction ID:*\n{alert_data.get('transaction_id', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Amount:*\n${alert_data.get('amount', 0)}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Probability:*\n{alert_data.get('probability', 0):.2%}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*User ID:*\n{alert_data.get('user_id', 'N/A')}"
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(webhook_url, json=slack_message)
        response.raise_for_status()
        
        logging.info(f"Slack alert sent for transaction {alert_data.get('transaction_id')}")
    
    def _log_alert(self, alert_data, alert_level, channels_used):
        """Log alert to MinIO for audit purposes"""
        try:
            alert_log = {
                "transaction_id": alert_data.get('transaction_id'),
                "alert_level": alert_level,
                "probability": alert_data.get('probability'),
                "amount": alert_data.get('amount'),
                "user_id": alert_data.get('user_id'),
                "channels_used": channels_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in MinIO
            log_data = json.dumps(alert_log).encode('utf-8')
            from io import BytesIO
            log_buffer = BytesIO(log_data)
            
            file_name = f"alerts/{datetime.now().strftime('%Y/%m/%d')}/{alert_data.get('transaction_id')}.json"
            minio_client.put_object(
                'fraud-data',
                file_name,
                log_buffer,
                len(log_data)
            )
            
        except Exception as e:
            logging.error(f"Failed to log alert: {e}")

alert_manager = AlertManager(config)

@app.route('/api/alerts/fraud', methods=['POST'])
def send_fraud_alert():
    try:
        alert_data = request.json
        if not alert_data:
            return jsonify({"error": "No alert data provided"}), 400
        
        result = alert_manager.send_alert(alert_data)
        
        return jsonify({
            "status": "success",
            "alert_id": alert_data.get('transaction_id'),
            "result": result
        })
    
    except Exception as e:
        logging.error(f"Alert sending error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "alert-service",
        "timestamp": datetime.now().isoformat(),
        "email_enabled": alert_manager.email_enabled,
        "slack_enabled": alert_manager.slack_enabled
    })

@app.route('/api/alerts/config', methods=['GET'])
def get_config():
    """Get current alert configuration (without secrets)"""
    safe_config = {
        "email_enabled": config['alerts']['email']['enabled'],
        "slack_enabled": config['alerts']['slack']['enabled'],
        "thresholds": config['alerts']['thresholds'],
        "recipients_count": len(config['alerts']['email']['recipients'])
    }
    return jsonify(safe_config)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
