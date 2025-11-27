#!/bin/bash
# scripts/deploy-complete.sh

echo "Starting Complete FraudShield Deployment..."

# Create namespace
kubectl create namespace fraudshield || true

# Deploy dependencies first
echo "Deploying MongoDB..."
kubectl apply -f kubernetes/mongo.yaml

echo "Deploying MinIO..."
kubectl apply -f kubernetes/minio.yaml

# Wait for dependencies
echo "Waiting for dependencies to be ready..."
kubectl wait --for=condition=ready pod -l app=mongodb -n fraudshield --timeout=60s
kubectl wait --for=condition=ready pod -l app=minio -n fraudshield --timeout=60s

# Build and push images
echo "Building Docker images..."
docker build -t fraudshield/data-service:latest data-service/
docker build -t fraudshield/feature-service:latest feature-service/
docker build -t fraudshield/model-service:latest model-service/
docker build -t fraudshield/alert-service:latest alert-service/

# Deploy microservices
echo "Deploying microservices..."
kubectl apply -f kubernetes/data-service.yaml
kubectl apply -f kubernetes/feature-service.yaml
kubectl apply -f kubernetes/model-service.yaml
kubectl apply -f kubernetes/alert-service.yaml

# Wait for services
echo "Waiting for services to be ready..."
kubectl wait --for=condition=ready pod -l app=data-service -n fraudshield --timeout=60s
kubectl wait --for=condition=ready pod -l app=feature-service -n fraudshield --timeout=60s
kubectl wait --for=condition=ready pod -l app=model-service -n fraudshield --timeout=60s
kubectl wait --for=condition=ready pod -l app=alert-service -n fraudshield --timeout=60s

echo "✅ Deployment completed successfully!"
echo ""
echo "📊 Services Status:"
kubectl get pods -n fraudshield

echo ""
echo "🌐 Service Endpoints:"
echo "Data Service: http://data-service.fraudshield.svc.cluster.local"
echo "Feature Service: http://feature-service.fraudshield.svc.cluster.local"
echo "Model Service: http://model-service.fraudshield.svc.cluster.local"
echo "Alert Service: http://alert-service.fraudshield.svc.cluster.local"
