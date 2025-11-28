#!/bin/bash
# Dissertation/scripts/train-model.sh
echo "Starting model training..."

# Trigger training
curl -X POST http://localhost:5005/api/training/train \
  -H "Content-Type: application/json" \
  -d '{"transaction_file": "train_transaction.csv", "identity_file": "train_identity.csv"}'

echo ""
echo "Training initiated! Check logs with: docker-compose logs -f training-service"