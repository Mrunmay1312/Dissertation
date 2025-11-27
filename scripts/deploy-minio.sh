#!/bin/bash
# Dissertation/scripts/deploy-minio.sh

echo "Deploying MinIO to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace fraudshield || true

# Deploy MinIO from the kubernetes directory
echo "Applying MinIO configuration..."
kubectl apply -f ../kubernetes/minio.yaml

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
kubectl wait --for=condition=ready pod -l app=minio -n fraudshield --timeout=300s

# Check MinIO status
echo "MinIO Pod Status:"
kubectl get pods -l app=minio -n fraudshield

# Wait for setup job to complete
echo "Waiting for MinIO setup job to complete..."
kubectl wait --for=condition=complete job/minio-setup -n fraudshield --timeout=180s

# Display connection information
echo ""
echo "🎉 MinIO Deployment Complete!"
echo ""
echo "📊 MinIO Access Information:"
echo "   Internal API URL: http://minio-service.fraudshield.svc.cluster.local:9000"
echo "   Internal Console URL: http://minio-service.fraudshield.svc.cluster.local:9001"
echo "   Access Key: minio"
echo "   Secret Key: minio123"
echo ""
echo "To access MinIO console, run:"
echo "  kubectl port-forward -n fraudshield svc/minio-service 9001:9001"
echo "Then open: http://localhost:9001"
