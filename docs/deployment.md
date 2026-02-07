# Deployment Guide

## Docker

### Build

```bash
docker build -t nibot .
```

The Dockerfile uses multi-stage build with `python:3.13-slim`, runs as non-root user `nibot`, and includes a health check.

### Run with docker-compose

1. Create your config file:

```bash
cp config.example.json config.json
# Edit config.json -- set channels, tools, etc.
# Do NOT put API keys in config.json -- use environment variables
```

2. Set provider keys as environment variables in `docker-compose.yml`:

```yaml
services:
  nibot:
    environment:
      - NIBOT_PROVIDERS__ANTHROPIC__API_KEY=sk-ant-xxx
      - NIBOT_CHANNELS__TELEGRAM__ENABLED=true
      - NIBOT_CHANNELS__TELEGRAM__TOKEN=bot:token
```

3. Start:

```bash
docker-compose up -d
```

4. Verify:

```bash
# Check container status
docker-compose ps

# Check health endpoint
curl http://localhost:9100/health

# View logs
docker-compose logs -f nibot
```

### Docker environment defaults

The Docker image sets these defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NIBOT_LOG__JSON_FORMAT` | `true` | JSON logs for log collectors |
| `NIBOT_LOG__LEVEL` | `INFO` | Log level |
| `NIBOT_HEALTH__ENABLED` | `true` | Enable health endpoint |
| `NIBOT_HEALTH__HOST` | `0.0.0.0` | Bind to all interfaces |

### Volumes

| Mount | Purpose |
|-------|---------|
| `./config.json:/home/nibot/.nibot/config.json:ro` | Config file (read-only) |
| `nibot-workspace` | Persistent workspace (sessions, skills, memory) |

## Mac Mini M4 Deployment

For running NiBot on Mac Mini M4 with local + cloud hybrid:

```bash
# Run the deployment helper
python scripts/deploy_mac.py
```

This script:
1. Detects macOS + Apple Silicon
2. Checks/installs Ollama
3. Pulls recommended local models
4. Generates a hybrid config (local Ollama + cloud API)
5. Generates `docker-compose.override.yml`

### Manual hybrid setup

```json
{
  "agent": {
    "model": "anthropic/claude-sonnet-4-5-20250929",
    "providerFallbackChain": ["anthropic", "ollama"]
  },
  "providers": {
    "anthropic": { "apiKey": "sk-ant-xxx" },
    "extras": {
      "ollama": { "apiBase": "http://localhost:11434" }
    }
  },
  "agents": {
    "coder": { "provider": "ollama", "model": "ollama/codellama" },
    "researcher": { "provider": "anthropic" }
  }
}
```

## Health Check

When `health.enabled=true`, NiBot serves `GET /health` on the configured port (default 9100):

```json
{
  "status": "ok",
  "uptime_seconds": 3600.5,
  "model": "anthropic/claude-sonnet-4-5-20250929",
  "channels": ["telegram"],
  "active_sessions": 5,
  "active_tasks": 0,
  "scheduler_jobs": 1
}
```

Status values:
- `ok` -- agent loop is running
- `degraded` -- agent loop is not running

## Webhook Server

For WeCom and API channels, enable the webhook server:

```bash
NIBOT_WEBHOOK__ENABLED=true
NIBOT_WEBHOOK__HOST=0.0.0.0
NIBOT_WEBHOOK__PORT=8080
```

Endpoints:
- `POST /webhook/wecom` -- WeCom callback
- `POST /api/chat` -- Synchronous API (JSON body: `{"content": "...", "sender_id": "..."}`)

## Logging

### JSON format (recommended for production)

```bash
NIBOT_LOG__JSON_FORMAT=true
```

### File output

```bash
NIBOT_LOG__FILE=/var/log/nibot/nibot.log
NIBOT_LOG__ROTATION="10 MB"
NIBOT_LOG__RETENTION="30 days"
```

### Debug mode

```bash
NIBOT_LOG__LEVEL=DEBUG
```
