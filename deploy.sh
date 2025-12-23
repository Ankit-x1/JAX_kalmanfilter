#!/bin/bash

# Professional Deployment Script for JAX Kalman Filter

set -e

PROJECT_NAME="jax-kalman-filter"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"

echo "=== JAX Kalman Filter Deployment ==="

# Build Docker image
echo "Building Docker image..."
docker build -t ${PROJECT_NAME}:latest .

# Tag for registry
echo "Tagging image for registry..."
docker tag ${PROJECT_NAME}:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}:latest

# Push to registry (optional)
if [ "$1" = "--push" ]; then
    echo "Pushing to registry..."
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:latest
fi

# Run production deployment
echo "Starting production deployment..."
docker-compose -f docker-compose.yml up -d kalman-filter

echo "=== Deployment Complete ==="
echo "Logs: docker logs -f kalman-filter"
echo "Stop: docker-compose down"
echo "Test: docker-compose --profile test up kalman-filter-test"