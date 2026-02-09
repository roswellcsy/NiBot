from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import NiBotConfig, _camel_to_snake, _convert_keys, load_config
from nibot.context import ContextBuilder
from nibot.memory import MemoryStore
from nibot.provider import LiteLLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import Session, SessionManager
from nibot.tools.exec_tool import ExecTool
from nibot.tools.file_tools import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool, _resolve_path
from nibot.types import Envelope, LLMResponse, SkillSpec, ToolCall


class DummyTool(Tool):
    def __init__(self, name: str = "dummy", fail: bool = False) -> None:
        self._name = name
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "dummy tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        if self._fail:
            raise RuntimeError("boom")
        return f"ok:{kwargs.get('x', '')}"


class DummySkills:
    def __init__(self, always: list[SkillSpec] | None = None, summary: str = "") -> None:
        self._always = always or []
        self._summary = summary

    def get_always_skills(self) -> list[SkillSpec]:
        return self._always

    def build_summary(self) -> str:
        return self._summary


class DummyChannel(BaseChannel):
    name = "dummy"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, envelope: Envelope) -> None:
        return None


class FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            return LLMResponse(content="")
        return self.responses.pop(0)


class DummyContextBuilder:
    def build(self, session: Session, current: Envelope) -> list[dict[str, Any]]:
        return [{"role": "system", "content": "sys"}]


# ===== types.py =====

def test_envelope_defaults() -> None:
    e = Envelope(channel="tg", chat_id="1", sender_id="u", content="hi")
    assert e.media == []
    assert e.metadata == {}
    assert e.timestamp is not None


def test_llm_response_has_tool_calls() -> None:
    r1 = LLMResponse(content="ok")
    assert r1.has_tool_calls is False
    r2 = LLMResponse(tool_calls=[ToolCall(id="1", name="t", arguments={})])
    assert r2.has_tool_calls is True


# ===== config.py =====

def test_camel_to_snake_and_convert_keys() -> None:
    assert _camel_to_snake("APIBase") == "api_base"
    assert _camel_to_snake("maxTokens") == "max_tokens"

    converted = _convert_keys({"Agent": {"maxTokens": 10}, "arr": [{"allowFrom": ["1"]}]})
    assert converted == {"agent": {"max_tokens": 10}, "arr": [{"allow_from": ["1"]}]}


def test_load_config_from_json_with_key_conversion(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps({"agent": {"maxTokens": 77}, "tools": {"execTimeout": 9}}),
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.agent.max_tokens == 77
    assert cfg.tools.exec_timeout == 9


def test_load_config_invalid_json_falls_back_defaults(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{bad", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.agent.name == "NiBot"


def test_load_config_default_values() -> None:
    cfg = NiBotConfig()
    assert cfg.agent.model == "anthropic/claude-opus-4-6"
    assert cfg.agent.max_iterations == 20
    assert cfg.tools.restrict_to_workspace is True


# ===== bus.py =====

@pytest.mark.asyncio
async def test_bus_inbound_publish_consume() -> None:
    bus = MessageBus()
    env = Envelope(channel="tg", chat_id="1", sender_id="u", content="hi")
    await bus.publish_inbound(env)
    got = await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    assert got.content == "hi"


@pytest.mark.asyncio
async def test_message_bus_dispatch_and_stop() -> None:
    bus = MessageBus()
    got: list[str] = []

    async def cb(env: Envelope) -> None:
        got.append(env.content)
        bus.stop()

    bus.subscribe_outbound("x", cb)
    task = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_outbound(Envelope(channel="x", chat_id="1", sender_id="u", content="hello"))
    await asyncio.wait_for(task, timeout=2)
    assert got == ["hello"]


@pytest.mark.asyncio
async def test_bus_dispatch_ignores_unsubscribed_channels() -> None:
    bus = MessageBus()
    got: list[str] = []

    async def cb(env: Envelope) -> None:
        got.append(env.content)
        bus.stop()

    bus.subscribe_outbound("x", cb)
    # Publish to channel "y" -- should not trigger callback
    await bus.publish_outbound(Envelope(channel="y", chat_id="1", sender_id="u", content="miss"))
    # Then publish to "x" to trigger stop
    await bus.publish_outbound(Envelope(channel="x", chat_id="1", sender_id="u", content="hit"))
    task = asyncio.create_task(bus.dispatch_outbound())
    await asyncio.wait_for(task, timeout=2)
    assert got == ["hit"]


# ===== registry.py =====

def test_registry_register_definitions_and_has() -> None:
    reg = ToolRegistry()
    t = DummyTool(name="alpha")
    reg.register(t)
    assert reg.has("alpha")
    assert not reg.has("beta")
    defs = reg.get_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "alpha"


def test_registry_deny_list_filtering() -> None:
    reg = ToolRegistry()
    reg.register(DummyTool(name="a"))
    reg.register(DummyTool(name="b"))
    filtered = reg.get_definitions(deny=["a"])
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "b"


@pytest.mark.asyncio
async def test_registry_execute_success() -> None:
    reg = ToolRegistry()
    reg.register(DummyTool(name="t"))
    result = await reg.execute("t", {"x": "val"})
    assert result.content == "ok:val"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_registry_execute_unknown_and_failure() -> None:
    reg = ToolRegistry()
    unknown = await reg.execute("missing", {})
    assert unknown.is_error is True
    assert "Unknown tool" in unknown.content

    reg.register(DummyTool(name="bad", fail=True))
    failed = await reg.execute("bad", {})
    assert failed.is_error is True
    assert failed.content.startswith("Error:")


# ===== session.py =====

def test_session_add_message_and_get_history() -> None:
    s = Session(key="k")
    s.add_message("user", "hello")
    s.add_message("assistant", "hi")
    history = s.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}


def test_session_get_history_max_messages() -> None:
    s = Session(key="k")
    for i in range(100):
        s.add_message("user", f"msg{i}")
    history = s.get_history(max_messages=10)
    assert len(history) == 10
    assert history[0]["content"] == "msg90"


def test_session_clear() -> None:
    s = Session(key="k")
    s.add_message("user", "x")
    s.clear()
    assert s.messages == []


def test_session_manager_save_load_delete(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    s = mgr.get_or_create("tg:123")
    s.add_message("user", "hello")
    s.add_message("assistant", "hi")
    mgr.save(s)

    mgr2 = SessionManager(tmp_path)
    s2 = mgr2.get_or_create("tg:123")
    assert [m["role"] for m in s2.messages] == ["user", "assistant"]
    assert s2.get_history()[-1]["content"] == "hi"

    mgr2.delete("tg:123")
    assert not mgr2._path_for("tg:123").exists()


def test_session_manager_corrupt_file_returns_new_session(tmp_path: Path) -> None:
    mgr = SessionManager(tmp_path)
    p = mgr._path_for("x")
    p.write_text("{not-json}\n", encoding="utf-8")
    s = mgr.get_or_create("x")
    assert s.key == "x"
    assert s.messages == []


# ===== memory.py =====

def test_memory_store_read_write_append_and_context(tmp_path: Path) -> None:
    mem = MemoryStore(tmp_path)
    mem.write_memory("fact one")
    mem.append_memory("fact two")
    mem.append_daily("today note")
    ctx = mem.get_context()
    assert "Long-term Memory" in ctx
    assert "fact one" in ctx
    assert "Today's Notes" in ctx


def test_memory_store_empty_context(tmp_path: Path) -> None:
    mem = MemoryStore(tmp_path)
    assert mem.get_context() == ""
    assert mem.read_memory() == ""
    assert mem.read_daily() == ""


# ===== context.py =====

def test_context_builder_includes_bootstrap_memory_and_skills(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "AGENTS.md").write_text("agent rules", encoding="utf-8")

    mem = MemoryStore(tmp_path / "memory")
    mem.write_memory("persistent fact")

    always = [SkillSpec(name="A", description="d", body="body", path="/x", always=True)]
    skills = DummySkills(always=always, summary="<skills></skills>")

    builder = ContextBuilder(config=NiBotConfig(), memory=mem, skills=skills, workspace=ws)
    session = Session(key="k")
    session.add_message("user", "old")
    env = Envelope(channel="telegram", chat_id="42", sender_id="u", content="new")
    msgs = builder.build(session=session, current=env)

    assert msgs[0]["role"] == "system"
    sys_prompt = msgs[0]["content"]
    assert "agent rules" in sys_prompt
    assert "persistent fact" in sys_prompt
    assert "## Skill: A" in sys_prompt
    assert "Current session: telegram:42" in sys_prompt
    assert msgs[-1]["role"] == "user"


def test_context_builder_user_content_with_media(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    image = tmp_path / "a.png"
    image.write_bytes(b"abc")

    builder = ContextBuilder(
        config=NiBotConfig(),
        memory=MemoryStore(tmp_path / "mem"),
        skills=DummySkills(),
        workspace=ws,
    )
    env = Envelope(channel="x", chat_id="1", sender_id="u", content="hello", media=[str(image)])
    content = builder._build_user_content(env)

    assert isinstance(content, list)
    assert content[-1]["type"] == "text"
    assert content[-1]["text"] == "hello"
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ===== file_tools.py =====

def test_resolve_path_workspace_enforcement(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    inside = _resolve_path("a.txt", ws)
    assert inside == (ws / "a.txt").resolve()

    with pytest.raises(PermissionError):
        _resolve_path("../outside.txt", ws)


def test_resolve_path_sibling_prefix_attack(tmp_path: Path) -> None:
    """Test the known bug: sibling dir sharing prefix can bypass check."""
    ws = tmp_path / "work"
    ws.mkdir()
    evil = tmp_path / "work_evil"
    evil.mkdir()
    (evil / "secret.txt").write_text("secret", encoding="utf-8")
    # This demonstrates the bug -- _resolve_path may not catch sibling paths
    # that share a string prefix with the workspace
    try:
        resolved = _resolve_path(str(evil / "secret.txt"), ws)
        # If we get here, the bug exists -- path was allowed
        assert False, f"Path traversal succeeded: {resolved}"
    except PermissionError:
        pass  # Fixed: correctly blocked


@pytest.mark.asyncio
async def test_file_tools_read_write_edit_list(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    read_tool = ReadFileTool(ws)
    write_tool = WriteFileTool(ws)
    edit_tool = EditFileTool(ws)
    list_tool = ListDirTool(ws)

    msg = await write_tool.execute(path="dir/f.txt", content="a\nb\nc\n")
    assert "Written" in msg

    read = await read_tool.execute(path="dir/f.txt", offset=1, limit=1)
    assert "lines 2-2" in read
    assert "2 | b" in read

    miss = await edit_tool.execute(path="dir/f.txt", old_text="zzz", new_text="x")
    assert "not found" in miss

    edited = await edit_tool.execute(path="dir/f.txt", old_text="b", new_text="B")
    assert "Edited" in edited

    listing = await list_tool.execute(path=".")
    assert "d dir" in listing


@pytest.mark.asyncio
async def test_read_file_not_found(tmp_path: Path) -> None:
    tool = ReadFileTool(tmp_path)
    result = await tool.execute(path="nonexistent.txt")
    assert "not found" in result.lower()


# ===== exec_tool.py =====

@pytest.mark.asyncio
async def test_exec_tool_blocks_dangerous_and_runs_safe_command(tmp_path: Path) -> None:
    tool = ExecTool(tmp_path, timeout=3)

    blocked = await tool.execute(command="rm -rf /")
    assert blocked.startswith("Blocked:")

    ok = await tool.execute(command="echo hello")
    assert "hello" in ok
    assert "[exit=0]" in ok


@pytest.mark.asyncio
async def test_exec_tool_timeout(tmp_path: Path) -> None:
    tool = ExecTool(tmp_path, timeout=1)
    cmd = f"\"{sys.executable}\" -c \"import time; time.sleep(5)\""
    result = await tool.execute(command=cmd, timeout=1)
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_exec_tool_multiple_dangerous_patterns(tmp_path: Path) -> None:
    tool = ExecTool(tmp_path, timeout=3)
    for cmd in ["format C:", "dd if=/dev/zero", "shutdown -h now", "mkfs.ext4 /dev/sda", "chmod -R 777 /"]:
        result = await tool.execute(command=cmd)
        assert result.startswith("Blocked:"), f"Should block: {cmd}"


# ===== channel.py =====

def test_channel_allowlist_empty_allows_all() -> None:
    bus = MessageBus()
    ch = DummyChannel(SimpleNamespace(allow_from=[]), bus)
    assert ch.is_allowed("anyone") is True


def test_channel_allowlist_exact_match() -> None:
    bus = MessageBus()
    ch = DummyChannel(SimpleNamespace(allow_from=["u1", "u2"]), bus)
    assert ch.is_allowed("u1") is True
    assert ch.is_allowed("u3") is False


def test_channel_allowlist_composite_id() -> None:
    bus = MessageBus()
    ch = DummyChannel(SimpleNamespace(allow_from=["team"]), bus)
    assert ch.is_allowed("x|team|y") is True
    assert ch.is_allowed("x|other|y") is False


@pytest.mark.asyncio
async def test_channel_handle_incoming_publishes_for_allowed() -> None:
    bus = MessageBus()
    ch = DummyChannel(SimpleNamespace(allow_from=["abc"]), bus)

    await ch._handle_incoming(sender_id="abc", chat_id="c1", content="hello")
    env = await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    assert env.channel == "dummy"
    assert env.chat_id == "c1"
    assert env.content == "hello"


@pytest.mark.asyncio
async def test_channel_handle_incoming_blocks_disallowed() -> None:
    bus = MessageBus()
    ch = DummyChannel(SimpleNamespace(allow_from=["abc"]), bus)

    await ch._handle_incoming(sender_id="blocked_user", chat_id="c1", content="hello")
    # Nothing should be in the queue
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(bus.consume_inbound(), timeout=0.2)


# ===== provider.py =====

def test_litellm_provider_configure_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)

    LiteLLMProvider(model="openrouter/xx", api_key="sk-or-abc")
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-abc"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    LiteLLMProvider(model="anthropic/claude-x", api_key="k2")
    assert os.environ.get("ANTHROPIC_API_KEY") == "k2"


def test_litellm_provider_parse_tool_calls() -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o")

    tc1 = SimpleNamespace(
        id="1",
        function=SimpleNamespace(name="t1", arguments='{"x":"y"}'),
    )
    tc2 = SimpleNamespace(
        id="2",
        function=SimpleNamespace(name="t2", arguments="{bad-json"),
    )
    msg = SimpleNamespace(content="resp", tool_calls=[tc1, tc2])
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    resp = SimpleNamespace(choices=[choice], usage={"prompt_tokens": 1, "completion_tokens": 2})

    parsed = provider._parse(resp)
    assert parsed.content == "resp"
    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0].arguments == {"x": "y"}
    assert parsed.tool_calls[1].arguments == {"raw": "{bad-json"}


# ===== agent.py =====

@pytest.mark.asyncio
async def test_agent_loop_process_without_tools(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = FakeProvider([LLMResponse(content="final")])
    reg = ToolRegistry()
    sessions = SessionManager(tmp_path / "sessions")
    cfg = NiBotConfig()
    loop = AgentLoop(bus, provider, reg, sessions, DummyContextBuilder(), cfg)

    env = Envelope(channel="tg", chat_id="1", sender_id="u", content="q")
    out = await loop._process(env)

    assert out.sender_id == "assistant"
    assert out.content == "final"

    saved = sessions.get_or_create("tg:1")
    assert saved.messages[-1]["role"] == "assistant"
    assert saved.messages[-1]["content"] == "final"


@pytest.mark.asyncio
async def test_agent_loop_process_with_tool_call(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = FakeProvider(
        [
            LLMResponse(
                content="thinking",
                tool_calls=[ToolCall(id="t1", name="dummy", arguments={"x": "v"})],
            ),
            LLMResponse(content="done"),
        ]
    )
    reg = ToolRegistry()
    reg.register(DummyTool(name="dummy"))
    sessions = SessionManager(tmp_path / "sessions")
    cfg = NiBotConfig()
    loop = AgentLoop(bus, provider, reg, sessions, DummyContextBuilder(), cfg)

    out = await loop._process(Envelope(channel="tg", chat_id="2", sender_id="u", content="q"))
    assert out.content == "done"
    assert len(provider.calls) == 2
    assert any(m.get("role") == "tool" and m.get("content") == "ok:v" for m in provider.calls[-1])


@pytest.mark.asyncio
async def test_agent_loop_max_iterations(tmp_path: Path) -> None:
    """Agent should stop after max_iterations even if LLM keeps requesting tools."""
    bus = MessageBus()
    # Provider always returns tool calls -- never stops
    infinite_responses = [
        LLMResponse(
            content="again",
            tool_calls=[ToolCall(id=f"t{i}", name="dummy", arguments={"x": str(i)})],
        )
        for i in range(100)
    ]
    provider = FakeProvider(infinite_responses)
    reg = ToolRegistry()
    reg.register(DummyTool(name="dummy"))
    sessions = SessionManager(tmp_path / "sessions")
    cfg = NiBotConfig()
    cfg.agent.max_iterations = 3
    loop = AgentLoop(bus, provider, reg, sessions, DummyContextBuilder(), cfg)

    out = await loop._process(Envelope(channel="tg", chat_id="3", sender_id="u", content="q"))
    # Should have called provider exactly max_iterations times
    assert len(provider.calls) == 3
    # Final content is fallback message because loop exhausted without a non-tool response
    assert "unable to complete" in out.content
