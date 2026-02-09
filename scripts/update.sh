#!/bin/bash
# NiBot update with rollback: pull, rebuild, restart, verify, rollback on failure.
set -e
cd "$(dirname "$0")/.."

echo "=== NiBot Update ==="

# Save current SHA for rollback
PREV_SHA=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current: $PREV_SHA ($BRANCH)"

# 1. Pull latest code
git pull origin "$BRANCH"
NEW_SHA=$(git rev-parse HEAD)
echo "Updated: $NEW_SHA"

if [ "$PREV_SHA" = "$NEW_SHA" ]; then
    echo "Already up to date. Rebuilding anyway..."
fi

# 2. Rebuild image
docker compose build

# 3. Restart with new image
docker compose up -d --force-recreate

# 4. Verify health (functional check, not just HTTP 200)
echo "Waiting for health check..."
HEALTHY=false
for i in $(seq 1 30); do
    RESPONSE=$(curl -sf http://localhost:9100/health 2>/dev/null || echo "")
    if [ -n "$RESPONSE" ]; then
        STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
        if [ "$STATUS" = "ok" ]; then
            HEALTHY=true
            break
        fi
    fi
    sleep 2
done

if $HEALTHY; then
    echo "NiBot is healthy! ($NEW_SHA)"
    docker compose logs --tail=5 nibot
    exit 0
fi

# 5. Rollback on failure
echo "DEPLOY FAILED -- rolling back to $PREV_SHA"
git checkout "$PREV_SHA"
docker compose build
docker compose up -d --force-recreate

echo "Rolled back to $PREV_SHA. Check logs:"
docker compose logs --tail=20 nibot
exit 1
