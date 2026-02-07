# NiBot

A self-evolving, multi-channel AI agent framework. The core stays lean; capabilities grow through skills; the system identifies its own weaknesses and improves itself.

## Features

- **4 channels**: Telegram, Feishu (Lark), WeCom (WeChat Work), HTTP API
- **23 built-in tools**: file I/O, code review, test runner, image generation, web search, git, scaffolding, and more
- **Sub-agent system**: Spawn specialized agents (coder, researcher, system, evolution) with isolated workspaces
- **Pipeline orchestration**: DAG-based multi-agent workflows with parallel execution
- **Self-evolution**: Metrics-driven skill creation, evaluation, and lifecycle management
- **MCP bridge**: Connect external MCP servers as native NiBot tools
- **Provider pool**: Multi-provider support with automatic failover (Anthropic, OpenAI, DeepSeek, Ollama, etc.)
- **Rate limiting**: Per-user and per-channel sliding window rate limiter
- **Health check**: HTTP `/health` endpoint for container orchestration
- **Structured logging**: JSON or text format, configurable level and file output
- **Docker ready**: Multi-stage Dockerfile with non-root user and health checks

## Quick Start

```bash
# Install
pip install -e ".[all,dev]"

# Create config with at least one provider key
cp config.example.json ~/.nibot/config.json
# Edit ~/.nibot/config.json -- set a provider API key

# Or use environment variables
export NIBOT_PROVIDERS__ANTHROPIC__API_KEY=sk-ant-xxx

# Run
nibot
```

See [docs/quickstart.md](docs/quickstart.md) for detailed setup instructions.

## Architecture

```
                    Channels (Telegram / Feishu / WeCom / API)
                                    |
                              MessageBus (async queues)
                                    |
                    AgentLoop (LLM + tool calling loop)
                          /         |          \
               ContextBuilder   ToolRegistry   SessionManager
              /    |    \          |    \            |
         Bootstrap Memory Skills  23 tools   JSONL persistence
                                   |
                            SubagentManager
                           /    |    |    \
                       coder researcher system evolution
                         |
                   WorktreeManager (git isolation)
```

**Key design principles**:
- Single-process, single-thread `asyncio` -- no locks needed
- `pydantic-settings` for config: JSON file + environment variable overrides
- LiteLLM as the universal LLM gateway
- JSONL for all persistence (sessions, evolution log, metrics)
- Zero mandatory external services -- runs standalone on a laptop

## Configuration

NiBot loads config from `~/.nibot/config.json` with environment variable overrides. Environment variables use the `NIBOT_` prefix with `__` as the nesting separator:

```bash
NIBOT_PROVIDERS__ANTHROPIC__API_KEY=sk-ant-xxx
NIBOT_CHANNELS__TELEGRAM__ENABLED=true
NIBOT_CHANNELS__TELEGRAM__TOKEN=bot:token
NIBOT_LOG__LEVEL=DEBUG
NIBOT_HEALTH__ENABLED=true
```

See [docs/configuration.md](docs/configuration.md) for the full reference.

## Docker

```bash
# Build
docker build -t nibot .

# Run with docker-compose
docker-compose up -d

# Check health
curl http://localhost:9100/health
```

See [docs/deployment.md](docs/deployment.md) for production deployment guide.

## Development

```bash
# Install dev dependencies
pip install -e ".[all,dev]"

# Run tests
pytest tests/ -v

# Project structure
nibot/
  app.py          # Composition root
  agent.py        # Main agent loop (LLM + tool calling)
  bus.py          # Async message bus
  config.py       # Pydantic settings schema
  context.py      # System prompt builder
  session.py      # Session manager (JSONL persistence)
  provider.py     # LiteLLM provider wrapper
  provider_pool.py # Multi-provider pool with failover
  registry.py     # Tool registry
  subagent.py     # Sub-agent manager
  worktree.py     # Git worktree isolation
  skills.py       # Skill loader (SKILL.md)
  scheduler.py    # Cron scheduler
  health.py       # Health check HTTP server
  metrics.py      # Session metrics + usage stats
  rate_limiter.py # Sliding window rate limiter
  channels/       # Telegram, Feishu, WeCom, API
  tools/          # 23 built-in tools
  web/            # Management web panel
tests/            # 419 tests
```

## License

MIT
