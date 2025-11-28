#!/bin/bash
# Dissertation/scripts/test-docker.sh
echo "Testing FraudShield services..."

# Test data service
echo "Testing Data Service..."
curl -X GET http://localhost:5001/api/health
echo ""

# Test feature service
echo "Testing Feature Service..."
curl -X GET http://localhost:5002/api/health
echo ""

# Test model service
echo "Testing Model Service..."
curl -X GET http://localhost:5003/api/health
echo ""

# Test alert service
echo "Testing Alert Service..."
curl -X GET http://localhost:5004/api/health
echo ""

echo "✅ All services are responding!"