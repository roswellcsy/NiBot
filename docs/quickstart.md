# Quick Start

## Requirements

- Python >= 3.11
- At least one LLM provider API key (or a local provider like Ollama)

## Installation

### From source

```bash
git clone https://github.com/your-org/nibot.git
cd nibot
pip install -e ".[all,dev]"
```

The `[all]` extra installs optional channel dependencies (Telegram, Feishu). The `[dev]` extra installs test tools (pytest, pytest-asyncio).

### With Docker

```bash
docker build -t nibot .
```

## Configuration

NiBot needs at least one LLM provider configured. The simplest way:

### Option A: Environment variable

```bash
export NIBOT_PROVIDERS__ANTHROPIC__API_KEY=sk-ant-xxx
nibot
```

### Option B: Config file

```bash
cp config.example.json ~/.nibot/config.json
```

Edit `~/.nibot/config.json` and fill in your provider API key:

```json
{
  "providers": {
    "anthropic": {
      "apiKey": "sk-ant-xxx"
    }
  }
}
```

Then run:

```bash
nibot
```

### Option C: Local provider (Ollama)

For keyless local providers, set `apiBase` instead of `apiKey`:

```bash
export NIBOT_PROVIDERS__EXTRAS__OLLAMA__API_BASE=http://localhost:11434
export NIBOT_AGENT__MODEL=ollama/llama3.2
nibot
```

## Connecting a Channel

### Telegram

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Get the bot token
3. Configure:

```bash
export NIBOT_CHANNELS__TELEGRAM__ENABLED=true
export NIBOT_CHANNELS__TELEGRAM__TOKEN=123456:ABC-DEF...
nibot
```

### Feishu (Lark)

1. Create an app at [Feishu Open Platform](https://open.feishu.cn)
2. Get app_id and app_secret
3. Configure:

```bash
export NIBOT_CHANNELS__FEISHU__ENABLED=true
export NIBOT_CHANNELS__FEISHU__APP_ID=cli_xxx
export NIBOT_CHANNELS__FEISHU__APP_SECRET=xxx
nibot
```

## Verify

After starting NiBot, send a message through your configured channel. You should receive an AI-generated response.

If health check is enabled (`NIBOT_HEALTH__ENABLED=true`), you can also verify:

```bash
curl http://localhost:9100/health
# {"status": "ok", "model": "anthropic/claude-sonnet-4-5-20250929", ...}
```

## Next Steps

- [Configuration Reference](configuration.md) -- all config options
- [Deployment Guide](deployment.md) -- Docker and Mac Mini M4 deployment
