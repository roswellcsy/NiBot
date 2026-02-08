"""Tests for VaultChannel -- file system watcher channel."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nibot.bus import MessageBus
from nibot.channels.vault import VaultChannel
from nibot.config import VaultChannelConfig
from nibot.types import Envelope


def _make_config(
    tmp_path: Path,
    *,
    tasks: dict[str, str] | None = None,
    notify_channel: str = "",
    notify_chat_id: str = "",
    enabled: bool = True,
) -> VaultChannelConfig:
    watch = tmp_path / "inbox"
    watch.mkdir(exist_ok=True)
    output = tmp_path / "processed"
    return VaultChannelConfig(
        enabled=enabled,
        watch_dir=str(watch),
        output_dir=str(output),
        poll_interval=1,
        notify_channel=notify_channel,
        notify_chat_id=notify_chat_id,
        tasks=tasks or {"summarize": "Summarize this article."},
    )


def _make_channel(
    tmp_path: Path,
    cfg: VaultChannelConfig | None = None,
    bus: MessageBus | None = None,
) -> tuple[VaultChannel, MessageBus]:
    bus = bus or MessageBus()
    cfg = cfg or _make_config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    ch = VaultChannel(cfg, bus, workspace=workspace)
    return ch, bus


# ---------- Core detection ----------


@pytest.mark.asyncio
async def test_new_file_detected(tmp_path: Path) -> None:
    """New .md file in a task subdirectory triggers inbound envelope."""
    ch, bus = _make_channel(tmp_path)
    bus.publish_inbound = AsyncMock()

    await ch.start()
    summarize_dir = Path(ch.config.watch_dir) / "summarize"
    summarize_dir.mkdir(exist_ok=True)
    (summarize_dir / "article.md").write_text("Hello world", encoding="utf-8")

    await asyncio.sleep(2.0)
    await ch.stop()

    bus.publish_inbound.assert_called_once()
    env: Envelope = bus.publish_inbound.call_args[0][0]
    assert env.channel == "vault"
    assert "Hello world" in env.content
    assert "Summarize this article." in env.content
    assert env.metadata["task_type"] == "summarize"


# ---------- Task routing ----------


@pytest.mark.asyncio
async def test_task_routing(tmp_path: Path) -> None:
    """Subdirectory name maps to correct prompt from config.tasks."""
    cfg = _make_config(tmp_path, tasks={
        "summarize": "PROMPT_A",
        "translate": "PROMPT_B",
    })
    ch, bus = _make_channel(tmp_path, cfg)
    bus.publish_inbound = AsyncMock()

    for task in ("summarize", "translate"):
        d = Path(cfg.watch_dir) / task
        d.mkdir(exist_ok=True)
        (d / "test.md").write_text(f"Content for {task}", encoding="utf-8")

    await ch.start()
    await asyncio.sleep(2.0)
    await ch.stop()

    assert bus.publish_inbound.call_count == 2
    contents = [call[0][0].content for call in bus.publish_inbound.call_args_list]
    assert any("PROMPT_A" in c for c in contents)
    assert any("PROMPT_B" in c for c in contents)


# ---------- Deduplication ----------


@pytest.mark.asyncio
async def test_already_processed_skipped(tmp_path: Path) -> None:
    """Same file is not processed twice."""
    ch, bus = _make_channel(tmp_path)
    bus.publish_inbound = AsyncMock()

    summarize_dir = Path(ch.config.watch_dir) / "summarize"
    summarize_dir.mkdir(exist_ok=True)
    (summarize_dir / "article.md").write_text("Content", encoding="utf-8")

    await ch.start()
    await asyncio.sleep(2.0)
    # File already processed -- wait another cycle
    await asyncio.sleep(2.0)
    await ch.stop()

    bus.publish_inbound.assert_called_once()


# ---------- State persistence ----------


@pytest.mark.asyncio
async def test_state_persistence(tmp_path: Path) -> None:
    """Processed files survive channel restart."""
    cfg = _make_config(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    # First run: process a file
    bus1 = MessageBus()
    bus1.publish_inbound = AsyncMock()
    ch1 = VaultChannel(cfg, bus1, workspace=workspace)

    summarize_dir = Path(cfg.watch_dir) / "summarize"
    summarize_dir.mkdir(exist_ok=True)
    (summarize_dir / "article.md").write_text("Content", encoding="utf-8")

    await ch1.start()
    await asyncio.sleep(2.0)
    await ch1.stop()
    assert bus1.publish_inbound.call_count == 1

    # Second run: same workspace, same file should be skipped
    bus2 = MessageBus()
    bus2.publish_inbound = AsyncMock()
    ch2 = VaultChannel(cfg, bus2, workspace=workspace)

    await ch2.start()
    await asyncio.sleep(2.0)
    await ch2.stop()
    assert bus2.publish_inbound.call_count == 0

    # Verify state file uses relative paths
    state_file = workspace / "vault_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(data["processed"]) == 1
    key = data["processed"][0]
    assert not Path(key).is_absolute(), f"State key should be relative, got: {key}"


# ---------- Output writeback ----------


@pytest.mark.asyncio
async def test_output_writeback(tmp_path: Path) -> None:
    """send() writes result to output_dir with correct structure."""
    ch, bus = _make_channel(tmp_path)
    await ch.start()

    env = Envelope(
        channel="vault",
        chat_id="summarize",
        sender_id="assistant",
        content="This is the summary.",
        metadata={"task_type": "summarize", "source_file": "/inbox/summarize/article.md"},
    )
    await ch.send(env)
    await ch.stop()

    out_file = Path(ch.config.output_dir) / "summarize" / "article.md"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "This is the summary."


# ---------- Notify channel ----------


@pytest.mark.asyncio
async def test_notify_channel(tmp_path: Path) -> None:
    """send() forwards truncated preview to notification channel."""
    cfg = _make_config(tmp_path, notify_channel="telegram", notify_chat_id="12345")
    ch, bus = _make_channel(tmp_path, cfg)
    bus.publish_outbound = AsyncMock()
    await ch.start()

    env = Envelope(
        channel="vault",
        chat_id="summarize",
        sender_id="assistant",
        content="Summary result here.",
        metadata={"task_type": "summarize", "source_file": "/inbox/summarize/article.md"},
    )
    await ch.send(env)
    await ch.stop()

    bus.publish_outbound.assert_called_once()
    forwarded: Envelope = bus.publish_outbound.call_args[0][0]
    assert forwarded.channel == "telegram"
    assert forwarded.chat_id == "12345"
    assert "Summary result here." in forwarded.content
    assert "[vault/summarize]" in forwarded.content


# ---------- Unknown task type ----------


@pytest.mark.asyncio
async def test_unknown_task_type(tmp_path: Path) -> None:
    """Files in unmapped subdirectory use raw content (no prompt prefix)."""
    ch, bus = _make_channel(tmp_path)
    bus.publish_inbound = AsyncMock()

    unknown_dir = Path(ch.config.watch_dir) / "unknown_task"
    unknown_dir.mkdir(exist_ok=True)
    (unknown_dir / "file.md").write_text("Raw content", encoding="utf-8")

    await ch.start()
    await asyncio.sleep(2.0)
    await ch.stop()

    bus.publish_inbound.assert_called_once()
    env: Envelope = bus.publish_inbound.call_args[0][0]
    assert env.content == "Raw content"
    assert env.metadata["task_type"] == "unknown_task"


# ---------- Disabled channel ----------


@pytest.mark.asyncio
async def test_disabled_channel(tmp_path: Path) -> None:
    """start() does nothing when watch_dir missing."""
    cfg = _make_config(tmp_path, enabled=False)
    cfg.enabled = False
    ch, bus = _make_channel(tmp_path, cfg)

    ch._watch_dir = tmp_path / "nonexistent"
    await ch.start()
    assert ch._watcher_task is None
    await ch.stop()


# ---------- Missing watch_dir ----------


@pytest.mark.asyncio
async def test_missing_watch_dir(tmp_path: Path) -> None:
    """start() logs warning and does not crash when watch_dir missing."""
    cfg = _make_config(tmp_path)
    ch, bus = _make_channel(tmp_path, cfg)

    ch._watch_dir = tmp_path / "does_not_exist"
    await ch.start()
    assert ch._watcher_task is None
    assert not ch._running
    await ch.stop()
