# NiBot Usage Guide

## Deployment Status

| Item | Value |
|------|-------|
| Host | Mac Mini M4 @ 192.168.5.55 |
| Health Check | http://192.168.5.55:9100/health |
| HTTP API | http://192.168.5.55:8080 |
| Web Panel | http://192.168.5.55:9200 |
| Main Model | `anthropic/claude-opus-4-6` |

## Channels

### HTTP API (enabled)

Internal integration and testing. Stateless request-response over HTTP.

```bash
# Simple conversation
curl -X POST http://192.168.5.55:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "chat_id": "test-001"}'

# Streaming response
curl -X POST http://192.168.5.55:8080/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Explain Linux kernel scheduling", "chat_id": "test-001"}'
```

### Feishu (pending credentials)

Requires `appId`, `appSecret`, `encryptKey` from Feishu Open Platform.

```yaml
# docker-compose.override.yml
environment:
  - NIBOT_CHANNELS__FEISHU__ENABLED=true
  - NIBOT_CHANNELS__FEISHU__APP_ID=cli_xxx
  - NIBOT_CHANNELS__FEISHU__APP_SECRET=xxx
  - NIBOT_CHANNELS__FEISHU__ENCRYPT_KEY=xxx
```

Webhook URL to configure in Feishu: `http://<host>:8080/webhook/feishu`

### WeCom (pending credentials)

Requires `corpId`, `secret`, `agentId` from WeCom admin console.

```yaml
# docker-compose.override.yml
environment:
  - NIBOT_CHANNELS__WECOM__ENABLED=true
  - NIBOT_CHANNELS__WECOM__CORP_ID=ww_xxx
  - NIBOT_CHANNELS__WECOM__SECRET=xxx
  - NIBOT_CHANNELS__WECOM__AGENT_ID=1000002
```

Webhook URL: `http://<host>:8080/webhook/wecom`

### Disabled Channels

- **Telegram** -- code retained, `enabled: false`
- **Discord** -- code retained, `enabled: false`
- **Vault** -- on-demand, `enabled: false`

## Core Capabilities

### Conversation
- Multi-turn memory with session isolation
- Streaming responses (SSE)
- Automatic context management (window trimming)

### File Operations
- `file_read` / `write_file` / `edit_file` -- workspace file CRUD
- `list_directory` -- directory listing
- `file_search` -- content search within workspace

### Code
- `code_review` -- automated code review
- `run_tests` -- test execution
- `git` -- Git operations (status, diff, commit, branch)
- `scaffold` -- project scaffolding

### Web
- `web_fetch` -- fetch and parse web pages
- `web_search` -- web search (requires API key)

### System
- `exec` -- shell command execution (timeout-protected)

### Management
- `config_get` / `config_set` -- runtime config inspection/modification
- `schedule_add` / `schedule_remove` -- cron job management
- `skill_create` / `skill_list` / `skill_delete` -- skill lifecycle

### Advanced
- `delegate` -- sub-agent delegation with role/goal/tools
- `pipeline` -- multi-step workflow orchestration
- MCP bridge -- connect external MCP servers as tools

## Web Management Panel

Read-only dashboard at port 9200.

| Page | Content |
|------|---------|
| Overview | Status, model info, active sessions, uptime |
| Sessions | Last 50 conversation sessions |
| Skills | Installed skills (view/delete) |
| Tasks | Sub-agent task list and status |
| Analytics | Aggregate metrics, error rates |

Access requires Bearer token:
```bash
curl http://192.168.5.55:9200/api/health \
  -H "Authorization: Bearer <AUTH_TOKEN>"
```

Configuration changes still require editing `config.json` or environment variables.

## Quick Examples

### Scheduled Job

```json
{
  "schedules": [
    {
      "name": "daily-summary",
      "cron": "0 9 * * *",
      "prompt": "Summarize yesterday's git commits",
      "channel": "scheduler",
      "tools": ["git", "file_read"]
    }
  ]
}
```

### Sub-Agent Delegation

```
Delegate to a code reviewer:
- Role: senior code reviewer
- Goal: review changes in src/auth/
- Tools: file_read, git
```

NiBot creates an isolated sub-agent with the specified role and tools, runs the task, and returns a summary.

### MCP Server Integration

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

## Next Steps

- [ ] Configure Feishu credentials and test webhook
- [ ] Configure WeCom credentials and test webhook
- [ ] Set up sub-agent model routing (Codex for coder/tester/reviewer)
- [ ] Evaluate agent self-generated skills (Pi/Ronacher approach)
