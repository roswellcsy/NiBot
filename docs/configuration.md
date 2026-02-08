# Configuration Reference

NiBot uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration. Config is loaded from a JSON file (`~/.nibot/config.json` by default) with environment variable overrides.

**Environment variables always win.** Use `NIBOT_` prefix with `__` as nesting separator.

## Agent

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `agent.name` | `NIBOT_AGENT__NAME` | `"NiBot"` | Bot display name |
| `agent.model` | `NIBOT_AGENT__MODEL` | `"anthropic/claude-sonnet-4-5-20250929"` | LiteLLM model identifier |
| `agent.maxTokens` | `NIBOT_AGENT__MAX_TOKENS` | `4096` | Max tokens per LLM response |
| `agent.temperature` | `NIBOT_AGENT__TEMPERATURE` | `0.7` | LLM temperature |
| `agent.maxIterations` | `NIBOT_AGENT__MAX_ITERATIONS` | `20` | Max tool-calling iterations per message |
| `agent.workspace` | `NIBOT_AGENT__WORKSPACE` | `"~/.nibot/workspace"` | Working directory for files, sessions, skills |
| `agent.contextWindow` | `NIBOT_AGENT__CONTEXT_WINDOW` | `128000` | Model context window size (tokens) |
| `agent.contextReserve` | `NIBOT_AGENT__CONTEXT_RESERVE` | `4096` | Reserved tokens for response |
| `agent.llmMaxRetries` | `NIBOT_AGENT__LLM_MAX_RETRIES` | `3` | LLM call retry count |
| `agent.autoEvolution` | `NIBOT_AGENT__AUTO_EVOLUTION` | `false` | Enable automatic self-evolution |
| `agent.providerFallbackChain` | `NIBOT_AGENT__PROVIDER_FALLBACK_CHAIN` | `[]` | Provider names to try in order on failure |

## Providers

At least one provider must have `apiKey` or `apiBase` set for NiBot to start.

```json
{
  "providers": {
    "anthropic": { "apiKey": "", "apiBase": "", "model": "" },
    "openai":    { "apiKey": "", "apiBase": "", "model": "" },
    "openrouter":{ "apiKey": "", "apiBase": "", "model": "" },
    "deepseek":  { "apiKey": "", "apiBase": "", "model": "" },
    "extras": {
      "ollama": { "apiKey": "", "apiBase": "http://localhost:11434" }
    }
  }
}
```

| Env Var | Example |
|---------|---------|
| `NIBOT_PROVIDERS__ANTHROPIC__API_KEY` | `sk-ant-xxx` |
| `NIBOT_PROVIDERS__OPENAI__API_KEY` | `sk-xxx` |
| `NIBOT_PROVIDERS__EXTRAS__OLLAMA__API_BASE` | `http://localhost:11434` |

## Channels

### Telegram

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `channels.telegram.enabled` | `NIBOT_CHANNELS__TELEGRAM__ENABLED` | `false` | Enable Telegram bot |
| `channels.telegram.token` | `NIBOT_CHANNELS__TELEGRAM__TOKEN` | `""` | Bot token from BotFather |
| `channels.telegram.allowFrom` | - | `[]` | Allowed user/chat IDs (empty = allow all) |

### Feishu (Lark)

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `channels.feishu.enabled` | `NIBOT_CHANNELS__FEISHU__ENABLED` | `false` | Enable Feishu bot |
| `channels.feishu.appId` | `NIBOT_CHANNELS__FEISHU__APP_ID` | `""` | App ID |
| `channels.feishu.appSecret` | `NIBOT_CHANNELS__FEISHU__APP_SECRET` | `""` | App Secret |
| `channels.feishu.encryptKey` | `NIBOT_CHANNELS__FEISHU__ENCRYPT_KEY` | `""` | Event encrypt key |

### WeCom (WeChat Work)

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `channels.wecom.enabled` | `NIBOT_CHANNELS__WECOM__ENABLED` | `false` | Enable WeCom bot |
| `channels.wecom.corpId` | `NIBOT_CHANNELS__WECOM__CORP_ID` | `""` | Corp ID |
| `channels.wecom.secret` | `NIBOT_CHANNELS__WECOM__SECRET` | `""` | App secret |
| `channels.wecom.agentId` | `NIBOT_CHANNELS__WECOM__AGENT_ID` | `""` | Agent ID |

### API

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `channels.api.enabled` | `NIBOT_CHANNELS__API__ENABLED` | `false` | Enable HTTP API channel |
| `channels.api.authTokens` | - | `[]` | Bearer tokens for authentication (empty = no auth) |

## Tools

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `tools.restrictToWorkspace` | `NIBOT_TOOLS__RESTRICT_TO_WORKSPACE` | `true` | Restrict file operations to workspace |
| `tools.execTimeout` | `NIBOT_TOOLS__EXEC_TIMEOUT` | `60` | Shell command timeout (seconds) |
| `tools.webSearchApiKey` | `NIBOT_TOOLS__WEB_SEARCH_API_KEY` | `""` | Web search API key |
| `tools.imageModel` | `NIBOT_TOOLS__IMAGE_MODEL` | `""` | Image generation model |
| `tools.pipelineMaxParallel` | `NIBOT_TOOLS__PIPELINE_MAX_PARALLEL` | `5` | Max parallel pipeline steps |

### MCP Servers

External MCP servers can be connected as NiBot tools:

```json
{
  "tools": {
    "mcpServers": {
      "my-server": {
        "command": "npx",
        "args": ["-y", "my-mcp-server"],
        "env": { "API_KEY": "xxx" }
      }
    }
  }
}
```

## Logging

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `log.level` | `NIBOT_LOG__LEVEL` | `"INFO"` | Log level: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `log.jsonFormat` | `NIBOT_LOG__JSON_FORMAT` | `false` | Output logs as JSON (for log collectors) |
| `log.file` | `NIBOT_LOG__FILE` | `""` | Log file path (empty = no file output) |
| `log.rotation` | `NIBOT_LOG__ROTATION` | `"10 MB"` | Log file rotation threshold |
| `log.retention` | `NIBOT_LOG__RETENTION` | `"7 days"` | Log file retention period |

## Health Check

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `health.enabled` | `NIBOT_HEALTH__ENABLED` | `false` | Enable HTTP health endpoint |
| `health.host` | `NIBOT_HEALTH__HOST` | `"127.0.0.1"` | Health server bind address |
| `health.port` | `NIBOT_HEALTH__PORT` | `9100` | Health server port |

## Rate Limiting

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `rateLimit.enabled` | `NIBOT_RATE_LIMIT__ENABLED` | `false` | Enable rate limiting |
| `rateLimit.perUserRpm` | `NIBOT_RATE_LIMIT__PER_USER_RPM` | `30` | Max requests per user per minute |
| `rateLimit.perChannelRpm` | `NIBOT_RATE_LIMIT__PER_CHANNEL_RPM` | `100` | Max requests per channel per minute |

## Webhook Server

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `webhook.enabled` | `NIBOT_WEBHOOK__ENABLED` | `false` | Enable webhook HTTP server |
| `webhook.host` | `NIBOT_WEBHOOK__HOST` | `"0.0.0.0"` | Bind address |
| `webhook.port` | `NIBOT_WEBHOOK__PORT` | `8080` | Port |

## Web Panel

| Field | Env Var | Default | Description |
|-------|---------|---------|-------------|
| `webPanel.enabled` | `NIBOT_WEB_PANEL__ENABLED` | `false` | Enable web management panel |
| `webPanel.host` | `NIBOT_WEB_PANEL__HOST` | `"127.0.0.1"` | Bind address |
| `webPanel.port` | `NIBOT_WEB_PANEL__PORT` | `9200` | Port |
| `webPanel.authToken` | `NIBOT_WEB_PANEL__AUTH_TOKEN` | `""` | Bearer token for API auth |

## Scheduled Jobs

```json
{
  "schedules": [
    {
      "id": "daily-report",
      "cron": "0 9 * * *",
      "prompt": "Generate a daily summary",
      "channel": "scheduler",
      "enabled": true
    }
  ]
}
```

## Sub-Agent Types

Override default sub-agent configurations:

```json
{
  "agents": {
    "coder": {
      "tools": ["file_read", "write_file", "edit_file", "exec", "git"],
      "maxIterations": 25,
      "workspaceMode": "worktree",
      "provider": "openai"
    },
    "researcher": {
      "tools": ["web_search", "web_fetch", "file_read"],
      "maxIterations": 15
    }
  }
}
```
