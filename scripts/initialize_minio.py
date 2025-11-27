# scripts/initialize_minio.py
from minio import Minio
from minio.error import S3Error
import time
import logging

logging.basicConfig(level=logging.INFO)

def initialize_minio():
    """Initialize MinIO with required buckets and setup"""
    
    # MinIO configuration
    minio_config = {
        'endpoint': 'minio-service:9000',
        'access_key': 'minio',
        'secret_key': 'minio123',
        'secure': False
    }
    
    # Buckets to create
    buckets = [
        'fraud-data',      # Raw transaction data
        'fraud-models',    # Trained models and artifacts
        'processed-data',  # Processed and feature-engineered data
        'backup',          # Backup files
        'logs'             # Application logs
    ]
    
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Initialize MinIO client
            minio_client = Minio(**minio_config)
            
            # Create buckets if they don't exist
            for bucket in buckets:
                if not minio_client.bucket_exists(bucket):
                    minio_client.make_bucket(bucket)
                    logging.info(f"✅ Created bucket: {bucket}")
                else:
                    logging.info(f"✅ Bucket already exists: {bucket}")
            
            # Set bucket policies
            try:
                # Make models bucket publicly readable (for model serving)
                minio_client.set_bucket_policy(
                    'fraud-models',
                    """{
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": ["s3:GetObject"],
                            "Resource": ["arn:aws:s3:::fraud-models/*"]
                        }
                    ]
                    }"""
                )
                logging.info("✅ Set public read policy for fraud-models bucket")
            except S3Error as e:
                logging.warning(f"Could not set bucket policy: {e}")
            
            logging.info("🎉 MinIO initialization completed successfully!")
            return True
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: MinIO connection failed - {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("❌ Failed to initialize MinIO after all retries")
                return False

if __name__ == "__main__":
    initialize_minio()
