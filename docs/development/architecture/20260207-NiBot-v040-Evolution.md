# NiBot v0.4.0 -- Gateway + Autonomous Evolution

## Overview

- **Date**: 2026-02-07
- **Goal**: Transform NiBot from monolithic agent to gateway orchestrator with specialized sub-agents, and add autonomous evolution capabilities.
- **Result**: 2341 production lines (+404), 107 tests (+28), zero breaking changes.

## What Changed

### Gateway Architecture
Main agent becomes an orchestrator. `gateway_tools` config field limits which tools the main AgentLoop can see. Sub-agents get their own tool whitelists via `AgentTypeConfig`.

- **Before**: AgentLoop had access to ALL tools. No role specialization.
- **After**: AgentLoop can be restricted to `[delegate, message, config, schedule, skill]`. Actual work delegated to typed sub-agents (coder, researcher, system, evolution).

### Typed Sub-Agents
`DelegateTool` (replaces `SpawnTool`) accepts `agent_type` parameter. `SubagentManager.spawn()` accepts `AgentTypeConfig` with:
- `tools`: whitelist (allow-list filtering via `registry.get_definitions(allow=...)`)
- `model`: override (forwarded to `provider.chat(model=...)`)
- `max_iterations`: per-type iteration limit
- `system_prompt`: additional system instructions

### Cron Scheduler
`SchedulerManager` fires scheduled tasks by publishing `Envelope` to inbound bus at cron-specified times. Pure message producer -- knows nothing about AgentLoop.

### Conversation-Based Management
Three admin tools in `admin_tools.py`:
- `ConfigTool`: get/set/list config with `_CONFIG_SAFE_FIELDS` security boundary
- `ScheduleTool`: add/remove/list cron jobs with persistence to `config.schedules`
- `SkillTool`: list/reload/get skills

### Skills Hot Reload
`SkillsLoader.reload()` clears and reloads all skills. Safe: no await between clear/load (atomic in single-threaded asyncio).

## File Changes

| File | Before | After | Delta |
|------|--------|-------|-------|
| config.py | 118 | 165 | +47 |
| registry.py | 77 | 81 | +4 |
| subagent.py | 97 | 116 | +19 |
| spawn_tool.py | 45 | 58 | +13 |
| agent.py | 126 | 130 | +4 |
| app.py | 134 | 143 | +9 |
| skills.py | 104 | 108 | +4 |
| bus.py | 49 | 51 | +2 |
| scheduler.py | NEW | 69 | +69 |
| admin_tools.py | NEW | 225 | +225 |
| **Production** | **1937** | **2341** | **+404** |
| test_v040_features.py | NEW | ~280 | +280 |
| **Tests** | **79** | **107** | **+28** |

## Backward Compatibility

- `gateway_tools` defaults to empty list = all tools visible (v0.3.0 behavior)
- `agents` defaults to empty dict = no typed sub-agents required
- `schedules` defaults to empty list = no cron jobs
- 79 original tests pass unchanged (2 mechanical renames: spawn -> delegate)

## Architecture Decisions

1. **Scheduler channel routing**: channel="scheduler" has no subscriber by default. Messages silently dropped with warning log. User-facing scheduled jobs must specify real channel (telegram, feishu).
2. **Config shared reference**: Pydantic v2 returns actual dict reference. ConfigTool modifies live config object -- DelegateTool sees changes immediately.
3. **File granularity**: 3 admin tools merged into single `admin_tools.py` (matches `file_tools.py` pattern).
4. **Skills reload safety**: No race condition -- single-threaded asyncio, `clear()` + `load_all()` with no await between them.

## New External Dependency

- `croniter` (pure Python, no C extensions) -- cron expression parsing

## Next Steps (v0.5.0)

- Web management UI (currently conversation-only)
- Scheduler `last_run` persistence for restart safety
- Multi-channel evolution (auto-detect conversation patterns per channel)
