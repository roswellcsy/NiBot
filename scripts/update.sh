#!/bin/bash
# One-click NiBot update: pull code, rebuild image, restart, verify health.
set -e
cd "$(dirname "$0")/.."

echo "=== NiBot Update ==="

# 1. Pull latest code (current branch)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git pull origin "$BRANCH"

# 2. Rebuild image
docker compose build

# 3. Restart with new image
docker compose up -d --force-recreate

# 4. Wait for health check
echo "Waiting for health check..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:9100/health > /dev/null 2>&1; then
        echo "NiBot is healthy!"
        docker compose logs --tail=5 nibot
        exit 0
    fi
    sleep 2
done

echo "WARNING: Health check did not pass within 60s"
docker compose logs --tail=20 nibot
exit 1
