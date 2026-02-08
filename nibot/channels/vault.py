"""Vault channel -- file system watcher for Obsidian Web Clipper integration."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import VaultChannelConfig
from nibot.log import logger
from nibot.types import Envelope

_MAX_FILE_SIZE = 512 * 1024  # 512 KB -- well within LLM context limits
_SAFE_NAME_RE = re.compile(r"^[\w\-. ]+$")  # alphanumeric, dash, dot, space, underscore


def _sanitize_name(name: str) -> str:
    """Ensure name contains no path separators or traversal components."""
    name = Path(name).name  # strip directory components
    if name in (".", "..") or not name:
        return "_invalid_"
    return name


class VaultChannel(BaseChannel):
    """Watch a directory for new .md files, route through AgentLoop, write results back."""

    name = "vault"

    def __init__(
        self, config: VaultChannelConfig, bus: MessageBus, *, workspace: Path,
    ) -> None:
        super().__init__(config, bus)
        self._watch_dir = Path(config.watch_dir).expanduser().resolve()
        self._output_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir else None
        self._poll_interval = max(config.poll_interval, 1)
        self._tasks_map = config.tasks
        self._notify_channel = config.notify_channel
        self._notify_chat_id = config.notify_chat_id
        self._state_path = workspace / "vault_state.json"
        self._processed: set[str] = set()  # relative paths from watch_dir
        self._watcher_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self._watch_dir.is_dir():
            logger.warning(f"Vault watch_dir does not exist: {self._watch_dir}")
            return
        self._load_state()
        self._running = True
        self._watcher_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Vault channel watching {self._watch_dir} (poll={self._poll_interval}s)")

    async def stop(self) -> None:
        self._running = False
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
        self._save_state()

    async def send(self, envelope: Envelope) -> None:
        meta = envelope.metadata or {}
        if meta.get("streaming") and not meta.get("stream_done"):
            return

        task_type = _sanitize_name(meta.get("task_type", ""))
        source_file = meta.get("source_file", "")
        filename = _sanitize_name(Path(source_file).name) if source_file else "output.md"

        if self._output_dir:
            if task_type:
                out_path = self._output_dir / task_type / filename
            else:
                out_path = self._output_dir / filename
            # Verify output stays within output_dir
            if not str(out_path.resolve()).startswith(str(self._output_dir)):
                logger.warning(f"Vault: output path escape blocked: {out_path}")
                return
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(envelope.content, encoding="utf-8")
            logger.info(f"Vault output written: {out_path}")

        if self._notify_channel and self._notify_chat_id:
            preview = envelope.content[:500]
            if len(envelope.content) > 500:
                preview += "\n..."
            label = f"[vault/{task_type}] {filename}" if task_type else f"[vault] {filename}"
            await self.bus.publish_outbound(Envelope(
                channel=self._notify_channel,
                chat_id=self._notify_chat_id,
                sender_id="vault",
                content=f"{label}\n\n{preview}",
            ))

    async def _watch_loop(self) -> None:
        while self._running:
            try:
                pending = await asyncio.to_thread(self._scan)
                for path, task_type in pending:
                    await self._process_file(path, task_type)
            except Exception as e:
                logger.error(f"Vault scan error: {e}")
            await asyncio.sleep(self._poll_interval)

    def _scan(self) -> list[tuple[Path, str]]:
        """Collect new .md files. Pure I/O, no async -- runs in thread."""
        pending: list[tuple[Path, str]] = []

        try:
            entries = sorted(self._watch_dir.iterdir())
        except OSError as e:
            logger.error(f"Vault: cannot list watch_dir: {e}")
            return pending

        for sub in entries:
            if sub.is_symlink() or not sub.is_dir():
                continue
            task_type = sub.name
            try:
                for md in sorted(sub.glob("*.md")):
                    if md.is_symlink():
                        continue
                    rel_key = str(md.relative_to(self._watch_dir))
                    if rel_key not in self._processed:
                        pending.append((md, task_type))
            except OSError as e:
                logger.warning(f"Vault: cannot scan {sub}: {e}")

        # Scan root-level .md files (no task type)
        try:
            for md in sorted(self._watch_dir.glob("*.md")):
                if md.is_symlink():
                    continue
                rel_key = str(md.relative_to(self._watch_dir))
                if rel_key not in self._processed:
                    pending.append((md, ""))
        except OSError as e:
            logger.warning(f"Vault: cannot scan root: {e}")

        return pending

    async def _process_file(self, path: Path, task_type: str) -> None:
        rel_key = str(path.relative_to(self._watch_dir))

        # Size check
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size > _MAX_FILE_SIZE:
            logger.warning(f"Vault: skipping oversized file ({size} bytes): {path.name}")
            self._processed.add(rel_key)
            self._save_state()
            return

        try:
            content = await asyncio.to_thread(path.read_text, "utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Vault: cannot read {path.name}: {e}")
            self._processed.add(rel_key)
            self._save_state()
            return

        if not content.strip():
            self._processed.add(rel_key)
            self._save_state()
            return

        prompt = self._tasks_map.get(task_type, "") if task_type else ""
        if prompt:
            full_content = f"[task: {task_type}]\n{prompt}\n\n---\n\n{content}"
        else:
            full_content = content

        meta: dict[str, Any] = {"source_file": str(path), "task_type": task_type}
        await self._handle_incoming(
            sender_id="vault",
            chat_id=task_type or "default",
            content=full_content,
            metadata=meta,
        )
        self._processed.add(rel_key)
        self._save_state()
        logger.info(f"Vault: queued {path.name} (task={task_type or 'raw'})")

    def _load_state(self) -> None:
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._processed = set(data.get("processed", []))
            except (json.JSONDecodeError, OSError):
                self._processed = set()

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"processed": sorted(self._processed)}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(self._state_path))
        except OSError as e:
            logger.error(f"Vault: failed to save state: {e}")
