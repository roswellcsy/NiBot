#!/bin/bash
# Standalone Docker run for NiBot (no compose).
# Mount paths follow Dockerfile CMD expectations.
set -e
cd "$(dirname "$0")/.."

CONTAINER_NAME="${1:-nibot}"
IMAGE="${2:-nibot:latest}"

# Stop existing
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p 8080:8080 \
    -p 9100:9100 \
    -p 9200:9200 \
    -v "$(pwd)/config.json:/home/nibot/.nibot/config.json:ro" \
    -v nibot-workspace:/home/nibot/.nibot/workspace \
    ${ENV_FILE:+--env-file "$ENV_FILE"} \
    "$IMAGE"

echo "NiBot started ($CONTAINER_NAME)."
echo "  Health:    http://localhost:9100/health"
echo "  API:       http://localhost:8080"
echo "  Web panel: http://localhost:9200"
