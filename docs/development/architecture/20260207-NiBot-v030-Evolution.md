# NiBot v0.3.0 Evolution Record

**Date**: 2026-02-07
**Goal**: Fix 1 critical bug, 3 concurrency/platform bugs, 2 security vulnerabilities
**Result**: All 7 items completed. 79 tests passing (63 old + 16 new).

## Summary

| Metric | v0.2.0 | v0.3.0 | Delta |
|--------|--------|--------|-------|
| Production lines | 1902 | 1937 | +35 |
| Test lines | 1002 | 1331 | +329 |
| Tests | 63 | 79 | +16 |
| Files modified | -- | 9 | -- |

## Phase 1: Bug Fixes

### 1.1 Subagent Result Delivery (P0)
- **Problem**: `subagent.py:86` used `publish_inbound(channel="system")` -- results entered inbound queue, got processed by AgentLoop creating fake "system:..." sessions, final response published to outbound "system" channel with zero subscribers. Users NEVER received subagent results.
- **Fix**: Changed to `publish_outbound()` with original channel/chat_id. Subagent already ran LLM+tools -- it just needs to deliver the final result directly.
- **Files**: `subagent.py`

### 1.2 SpawnTool Shared Mutable State
- **Problem**: `_origin_channel`/`_origin_chat_id` instance attributes overwritten by `receive_context()`. Concurrent requests would clobber each other's context.
- **Fix**: Eliminated instance state. Registry passes `_tool_ctx` through `execute(**kwargs)`. SpawnTool extracts context from kwargs per-call. Removed `receive_context()` call from registry.
- **Files**: `registry.py`, `spawn_tool.py`

### 1.3 Concurrent Processing + Session Locking
- **Problem**: `agent.py` while-loop processed messages serially. One slow LLM call blocked all channels.
- **Fix**: `asyncio.create_task()` for concurrent dispatch. Added `SessionManager.lock_for(key)` returning per-session `asyncio.Lock`. `_process()` wraps entire session access in lock.
- **Files**: `agent.py`, `session.py`

### 1.4 Telegram Message Chunking
- **Problem**: Telegram API limits messages to 4096 chars. Longer content caused `send_message` to fail, caught and swallowed by error handler. Users received nothing.
- **Fix**: Split content into 4096-char chunks, send each sequentially.
- **Files**: `channels/telegram.py`

## Phase 2: Security Fixes

### 2.1 WebFetchTool SSRF Protection
- **Problem**: LLM could instruct WebFetchTool to fetch internal network URLs (169.254.169.254 metadata, localhost, 10.x/172.16.x/192.168.x private ranges).
- **Fix**: Added `_is_private_url()` using `ipaddress` stdlib. Resolves hostname, checks all returned IPs against private/loopback/reserved/link-local ranges. Blocks before HTTP request.
- **Files**: `tools/web_tools.py`

### 2.2 Provider Error Sanitization
- **Problem**: `LLMResponse(content=f"LLM error: {last_error}")` could expose API keys or sensitive request details in error messages returned to users.
- **Fix**: Error content now only contains exception type name: `f"LLM error: {type(e).__name__}"`. Full details remain in logger.error().
- **Files**: `provider.py`

## Phase 3: Tests (+16)

| Test Class | Count | What it tests |
|-----------|-------|---------------|
| TestSubagentDelivery | 2 | Results reach correct channel; errors too |
| TestSpawnToolStateless | 3 | Context from kwargs; no instance state; works without context |
| TestConcurrentProcessing | 3 | lock_for() behavior; create_task usage; lock serialization |
| TestTelegramChunking | 2 | Long message split; short message not split |
| TestSSRFProtection | 5 | localhost, private IP, link-local, empty hostname blocked; public URL allowed |
| TestErrorSanitization | 1 | Only type name exposed, not secret details |

## Backward Compatibility

All changes backward compatible:
- 63 original tests pass (2 assertions updated to match intentional v0.3.0 behavior changes: context via kwargs instead of receive_context, error type-only instead of full message)
- `Tool.receive_context()` base method still exists (no-op), no breaking change for subclasses
- `_tool_ctx` kwarg silently ignored by tools that don't use it
- Session lock_for() is additive -- no change to save/load/delete behavior

## What's Next (v0.4.0 candidates)

- Agent/Channel streaming integration (editMessage for Telegram, etc.)
- Request logging + token usage tracking
- Graceful shutdown queue drain (if needed)
- Observability dashboard (if needed)
