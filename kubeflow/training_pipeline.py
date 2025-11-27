# kubeflow/training_pipeline.py
import kfp
from kfp import dsl
from kfp.components import func_to_container_op
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from minio import Minio
import numpy as np

@func_to_container_op
def load_and_preprocess_data():
    """Load and preprocess transaction data"""
    # Generate synthetic fraud data for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'amount': np.random.exponential(100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'transaction_count_24h': np.random.poisson(5, n_samples),
        'avg_amount_7d': np.random.exponential(80, n_samples),
        'distance_from_home': np.random.exponential(10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic fraud labels
    fraud_prob = (
        0.1 * (df['amount'] > 500) +
        0.2 * (df['hour'].between(0, 5)) +
        0.15 * (df['distance_from_home'] > 50) +
        0.1 * (df['transaction_count_24h'] > 10) +
        np.random.random(n_samples) * 0.1
    )
    
    df['is_fraud'] = (fraud_prob > 0.5).astype(int)
    
    # Save to MinIO
    minio_client = Minio(
        "minio-service.kubeflow:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    
    df.to_csv('transaction_data.csv', index=False)
    minio_client.fput_object('fraud-data', 'training/transaction_data.csv', 'transaction_data.csv')
    
    return df.to_dict()

@func_to_container_op
def train_model(data_dict: dict) -> str:
    """Train fraud detection model"""
    df = pd.DataFrame(data_dict)
    
    # Prepare features and target
    features = ['amount', 'hour', 'day_of_week', 'transaction_count_24h', 'avg_amount_7d', 'distance_from_home']
    X = df[features]
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, '/tmp/fraud_model.pkl')
    
    # Save to MinIO
    minio_client = Minio(
        "minio-service.kubeflow:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    
    minio_client.fput_object('fraud-models', 'fraud_model.pkl', '/tmp/fraud_model.pkl')
    
    return f"Model trained with AUC: {auc:.4f}"

@func_to_container_op
def deploy_model(model_output: str):
    """Deploy model to model service"""
    print(f"Deploying model: {model_output}")
    
    # Download model from MinIO and deploy to model service
    minio_client = Minio(
        "minio-service.kubeflow:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    
    # This would typically involve updating the model service
    # For now, we'll just print the deployment
    print("Model deployment completed")

@dsl.pipeline(
    name='Fraud Detection Training Pipeline',
    description='End-to-end pipeline for fraud detection model training'
)
def fraud_training_pipeline():
    # Define pipeline steps
    load_data_op = load_and_preprocess_data()
    train_model_op = train_model(load_data_op.output)
    deploy_model_op = deploy_model(train_model_op.output)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(fraud_training_pipeline, 'fraud_training_pipeline.yaml')
