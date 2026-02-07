# NiBot v0.2.0 Evolution Record

**Date**: 2026-02-06
**Goal**: Fix 3 critical bugs, 4 architecture gaps, 3 production features
**Result**: All 11 items completed. 61 tests passing (39 old + 22 new).

## Summary

| Metric | v0.1.0 | v0.2.0 | Delta |
|--------|--------|--------|-------|
| Production lines | 1771 | 1902 | +131 |
| Test lines | 607 | 1002 | +395 |
| Tests | 39 | 61 | +22 |
| Files | 25 | 25+1 test | +1 |

## Phase 1: Critical Bug Fixes

### 1.1 ToolResult.call_id Propagation
- **Problem**: `ToolResult(call_id="")` everywhere -- data structure lied
- **Fix**: `registry.execute()` accepts `call_id` param, propagates to ToolResult
- **Files**: `registry.py`, `agent.py`, `subagent.py`

### 1.2 Session Complete Message Thread
- **Problem**: Only user+assistant saved, tool_calls/tool results lost
- **Fix**: `get_history()` returns full message dicts (minus timestamp); `agent.py` saves all new messages from LLM loop
- **Files**: `session.py`, `agent.py`

### 1.3 Error Notification to User
- **Problem**: Exceptions swallowed by `logger.error`, user sees nothing
- **Fix**: catch block publishes error Envelope to outbound bus
- **Files**: `agent.py`

### 1.4 max_iterations Exhaustion Notification
- **Problem**: Loop exhaustion -> empty content -> user gets blank message
- **Fix**: Fallback message "unable to complete the task within the allowed steps"
- **Files**: `agent.py`

## Phase 2: Architecture Improvements

### 2.1 ToolExecutionContext
- **Problem**: `agent.py` accessed `registry._tools` (private) for SpawnTool.set_origin() hack
- **Fix**: New `ToolContext` dataclass in `types.py`; `Tool.receive_context()` default method; registry passes ctx before execute; SpawnTool overrides receive_context()
- **Files**: `types.py`, `registry.py`, `agent.py`, `spawn_tool.py`

### 2.2 Config Environment Variable Priority
- **Problem**: `NiBotConfig(**file_data)` init kwargs overrode env vars
- **Fix**: `settings_customise_sources()` orders: env > init > file_secret
- **Files**: `config.py`

### 2.3 Token Budget Management
- **Problem**: Context builder blindly concatenated everything, long conversations overflow context window
- **Fix**: `_estimate_tokens()` with litellm fallback to 4-chars/token; `build()` fills history from newest backwards until budget exhausted
- **Files**: `context.py`, `config.py` (context_window, context_reserve)

### 2.4 BOOTSTRAP_FILES Configurable
- **Problem**: Hardcoded constant
- **Fix**: `AgentConfig.bootstrap_files` list, context builder reads from config
- **Files**: `config.py`, `context.py`

### 2.5 Cleanup
- Removed `_bus_ref()` -- passed `self.bus` directly
- Moved `_register_builtin_tools()` from `run()` to `__init__()` end
- **Files**: `app.py`

## Phase 3: Production Features

### 3.1 Provider Retry with Exponential Backoff
- `delay = base * 2^attempt`, configurable `llm_max_retries` and `llm_retry_base_delay`
- **Files**: `provider.py`, `config.py`, `app.py`

### 3.2 MessageBus Backpressure
- `MessageBus(maxsize=N)` -- configurable `bus_queue_maxsize` (0=unlimited, backward compatible)
- **Files**: `bus.py`, `config.py`, `app.py`

### 3.3 Streaming Response (Provider Layer)
- `LLMProvider.chat_stream()` default falls back to non-streaming
- `LiteLLMProvider.chat_stream()` real streaming (falls back when tools present)
- Agent/Channel layer unchanged -- streaming activation deferred to v0.3.0
- **Files**: `provider.py`

## New Config Fields

```json
{
  "agent": {
    "bootstrapFiles": ["IDENTITY.md", "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"],
    "contextWindow": 128000,
    "contextReserve": 4096,
    "llmMaxRetries": 3,
    "llmRetryBaseDelay": 1.0,
    "busQueueMaxsize": 0
  }
}
```

## Backward Compatibility

All changes backward compatible:
- New parameters have defaults matching v0.1.0 behavior
- `Tool.receive_context()` is a no-op by default
- `bus_queue_maxsize=0` means unlimited (same as v0.1.0)
- 39 original tests pass without modification (1 assertion updated to match intentional behavior change)

## What's Next (v0.3.0 candidates)

- Agent/Channel streaming integration (editMessage for Telegram, etc.)
- Middleware/hook pipeline
- Rate limiting per channel/user
- Observability (metrics, tracing)
