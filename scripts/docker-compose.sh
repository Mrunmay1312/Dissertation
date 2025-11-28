#!/bin/bash
# Dissertation/scripts/run-docker.sh
echo "Starting FraudShield with Docker Compose..."

# Build and start all services
docker-compose up --build -d

echo "Services are starting..."
echo ""
echo "📊 Service URLs:"
echo "  Data Service:      http://localhost:5001"
echo "  Feature Service:   http://localhost:5002"
echo "  Model Service:     http://localhost:5003"
echo "  Alert Service:     http://localhost:5004"
echo "  MinIO Console:     http://localhost:9001 (minio/minio123)"
echo "  MongoDB:           localhost:27017"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"