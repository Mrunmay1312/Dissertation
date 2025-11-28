# Dissertation/training-service/app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from minio import Minio
import warnings
import logging
from io import BytesIO
import os
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def get_minio_client():
    return Minio(
        os.getenv('MINIO_URL', 'minio:9000').replace('http://', ''),
        access_key=os.getenv('MINIO_ACCESS_KEY', 'minio'),
        secret_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),
        secure=False
    )

class FraudModelTrainer:
    def __init__(self):
        self.minio_client = get_minio_client()
        
    def load_ieee_data(self, transaction_file='transaction.csv', identity_file='identity.csv'):
        """Load and merge IEEE fraud detection datasets"""
        try:
            # Load from mounted volume first, then from MinIO
            local_transaction_path = f'/app/data/{transaction_file}'
            local_identity_path = f'/app/data/{identity_file}'
            
            if os.path.exists(local_transaction_path) and os.path.exists(local_identity_path):
                logging.info("Loading data from local files...")
                transaction_df = pd.read_csv(local_transaction_path)
                identity_df = pd.read_csv(local_identity_path)
            else:
                # Try to load from MinIO
                logging.info("Loading data from MinIO...")
                transaction_response = self.minio_client.get_object('fraud-data', transaction_file)
                transaction_df = pd.read_csv(BytesIO(transaction_response.read()))
                transaction_response.close()
                
                identity_response = self.minio_client.get_object('fraud-data', identity_file)
                identity_df = pd.read_csv(BytesIO(identity_response.read()))
                identity_response.close()
            
            # Merge datasets on TransactionID
            logging.info("Merging transaction and identity data...")
            merged_df = pd.merge(transaction_df, identity_df, on='TransactionID', how='left')
            
            # Save merged data to MinIO
            merged_csv = merged_df.to_csv(index=False).encode('utf-8')
            merged_buffer = BytesIO(merged_csv)
            self.minio_client.put_object('processed-data', 'merged_fraud_data.csv', merged_buffer, len(merged_csv))
            
            logging.info(f"Merged data shape: {merged_df.shape}")
            return merged_df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e
    
    def preprocess_data(self, df):
        """Preprocess the IEEE fraud detection data"""
        logging.info("Preprocessing data...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.create_features(df)
        
        # Select features for modeling
        feature_columns = self.select_features(df)
        
        # Prepare final dataset
        X = df[feature_columns]
        y = df['isFraud']
        
        logging.info(f"Final feature set: {len(feature_columns)} features")
        logging.info(f"Fraud rate: {y.mean():.4f}")
        
        return X, y, feature_columns
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing', inplace=True)
        
        return df
    
    def create_features(self, df):
        """Create new features from existing ones"""
        # Transaction amount features
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            df['TransactionAmt_decimal'] = df['TransactionAmt'] - np.floor(df['TransactionAmt'])
        
        # Time-based features
        if 'TransactionDT' in df.columns:
            df['TransactionHour'] = (df['TransactionDT'] // 3600) % 24
            df['TransactionDay'] = (df['TransactionDT'] // 86400) % 7
            df['TransactionHour_sin'] = np.sin(2 * np.pi * df['TransactionHour'] / 24)
            df['TransactionHour_cos'] = np.cos(2 * np.pi * df['TransactionHour'] / 24)
        
        # Email domain features
        if 'P_emaildomain' in df.columns:
            df['P_emaildomain_simple'] = df['P_emaildomain'].str.split('.').str[0]
        
        # Card features
        card_cols = [col for col in df.columns if 'card' in col.lower()]
        if card_cols:
            df['card_features_count'] = df[card_cols].notna().sum(axis=1)
        
        return df
    
    def select_features(self, df):
        """Select relevant features for modeling"""
        # Remove columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        low_missing_cols = missing_ratio[missing_ratio < 0.3].index
        
        # Select numerical features with some variance
        numerical_cols = df[low_missing_cols].select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'isFraud']
        
        # Remove constant columns
        high_variance_cols = []
        for col in numerical_cols:
            if df[col].nunique() > 1:
                high_variance_cols.append(col)
        
        # Select top 50 features by correlation with target
        if len(high_variance_cols) > 50:
            correlations = df[high_variance_cols].corrwith(df['isFraud']).abs().sort_values(ascending=False)
            high_variance_cols = correlations.head(50).index.tolist()
        
        # Always include target
        if 'isFraud' not in high_variance_cols:
            high_variance_cols.append('isFraud')
        
        return high_variance_cols
    
    def train_models(self, X, y, feature_names):
        """Train multiple models and select the best one"""
        logging.info("Training models...")
        
        # Handle class imbalance
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'xgboost': None  # Will use XGBoost if available
        }
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            logging.warning("XGBoost not available, skipping...")
            del models['xgboost']
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            if model is None:
                continue
                
            logging.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            results[name] = {
                'auc': auc,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logging.info(f"{name} AUC: {auc:.4f}")
            
            if auc > best_score:
                best_score = auc
                best_model = name
        
        # Save best model and scaler
        best_model_obj = results[best_model]['model']
        
        # Save to MinIO
        model_bytes = BytesIO()
        joblib.dump(best_model_obj, model_bytes)
        model_bytes.seek(0)
        self.minio_client.put_object('fraud-models', 'best_fraud_model.pkl', model_bytes, len(model_bytes.getvalue()))
        
        scaler_bytes = BytesIO()
        joblib.dump(scaler, scaler_bytes)
        scaler_bytes.seek(0)
        self.minio_client.put_object('fraud-models', 'scaler.pkl', scaler_bytes, len(scaler_bytes.getvalue()))
        
        # Save feature names
        feature_info = {'feature_names': feature_names}
        feature_bytes = BytesIO()
        joblib.dump(feature_info, feature_bytes)
        feature_bytes.seek(0)
        self.minio_client.put_object('fraud-models', 'feature_info.pkl', feature_bytes, len(feature_bytes.getvalue()))
        
        # Generate evaluation report
        evaluation_report = self.generate_evaluation_report(results, X_test, y_test)
        
        logging.info(f"Best model: {best_model} with AUC: {best_score:.4f}")
        return evaluation_report
    
    def generate_evaluation_report(self, results, X_test, y_test):
        """Generate comprehensive evaluation report"""
        report = {
            'models_trained': list(results.keys()),
            'best_model': max(results.keys(), key=lambda x: results[x]['auc']),
            'test_set_size': len(X_test),
            'fraud_rate_test': y_test.mean(),
            'training_time': datetime.now().isoformat()
        }
        
        for model_name, result in results.items():
            report[model_name] = {
                'auc': float(result['auc']),
                'confusion_matrix': confusion_matrix(y_test, result['predictions']).tolist(),
                'classification_report': classification_report(y_test, result['predictions'], output_dict=True)
            }
        
        # Save report to MinIO
        report_json = json.dumps(report).encode('utf-8')
        report_buffer = BytesIO(report_json)
        self.minio_client.put_object('fraud-models', 'training_report.json', report_buffer, len(report_json))
        
        return report

trainer = FraudModelTrainer()

@app.route('/api/training/train', methods=['POST'])
def train_model():
    """Train fraud detection model"""
    try:
        data = request.json or {}
        transaction_file = data.get('transaction_file', 'transaction.csv')
        identity_file = data.get('identity_file', 'identity.csv')
        
        logging.info("Starting model training...")
        
        # Load data
        df = trainer.load_ieee_data(transaction_file, identity_file)
        
        # Preprocess data
        X, y, feature_names = trainer.preprocess_data(df)
        
        # Train models
        results = trainer.train_models(X, y, feature_names)
        
        return jsonify({
            "status": "success",
            "message": "Model training completed",
            "results": results
        })
    
    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/training/upload', methods=['POST'])
def upload_training_data():
    """Upload training data to MinIO"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        file_type = request.form.get('type', 'transaction')  # transaction or identity
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save to MinIO
        file_bytes = file.read()
        file_buffer = BytesIO(file_bytes)
        
        file_name = f"{file_type}.csv"
        trainer.minio_client.put_object('fraud-data', file_name, file_buffer, len(file_bytes))
        
        return jsonify({
            "status": "success",
            "message": f"File {file_name} uploaded successfully",
            "size": len(file_bytes)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/training/status', methods=['GET'])
def training_status():
    """Get training status and available models"""
    try:
        # Check if model exists
        try:
            trainer.minio_client.stat_object('fraud-models', 'best_fraud_model.pkl')
            model_exists = True
        except:
            model_exists = False
        
        # List available datasets
        datasets = []
        for obj in trainer.minio_client.list_objects('fraud-data'):
            datasets.append(obj.object_name)
        
        return jsonify({
            "model_available": model_exists,
            "datasets": datasets,
            "status": "ready"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "training-service"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)