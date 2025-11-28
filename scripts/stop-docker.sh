#!/bin/bash
# Dissertation/scripts/stop-docker.sh
echo "Stopping FraudShield services..."
docker-compose down
echo "All services stopped."