#!/bin/bash
# scripts/verify-minio.sh

echo "Verifying MinIO deployment..."

# Check if MinIO pod is running
MINIO_POD=$(kubectl get pods -n fraudshield -l app=minio -o jsonpath='{.items[0].metadata.name}')
if [ -z "$MINIO_POD" ]; then
    echo "❌ MinIO pod not found"
    exit 1
fi

echo "✅ MinIO pod: $MINIO_POD"

# Check pod status
POD_STATUS=$(kubectl get pod -n fraudshield $MINIO_POD -o jsonpath='{.status.phase}')
if [ "$POD_STATUS" != "Running" ]; then
    echo "❌ MinIO pod is not running. Status: $POD_STATUS"
    exit 1
fi

echo "✅ MinIO pod status: $POD_STATUS"

# Test MinIO connectivity
echo "Testing MinIO connectivity..."
kubectl exec -n fraudshield $MINIO_POD -- curl -s http://localhost:9000/minio/health/live > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ MinIO health check passed"
else
    echo "❌ MinIO health check failed"
    exit 1
fi

# List buckets using MinIO client
echo "Listing MinIO buckets..."
kubectl exec -n fraudshield $MINIO_POD -- /bin/sh -c "
MC_PATH=/opt/bin/mc
if [ -f \$MC_PATH ]; then
    \$MC_PATH alias set myminio http://localhost:9000 minio minio123
    \$MC_PATH ls myminio
else
    echo 'MC client not found in container'
fi
"

echo ""
echo "🎉 MinIO verification completed successfully!"
echo ""
echo "Quick access commands:"
echo "  kubectl port-forward -n fraudshield svc/minio-service 9001:9001  # Access console"
echo "  kubectl logs -n fraudshield $MINIO_POD                          # View logs"
echo "  kubectl exec -n fraudshield $MINIO_POD -- /bin/sh               # Access container"
