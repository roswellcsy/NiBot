"""File system tools -- read, write, edit, list_dir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nibot.registry import Tool


def _resolve_path(path: str, workspace: Path, restrict: bool = True) -> Path:
    """Resolve path. When restrict=True, deny access outside workspace."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    if restrict and not p.is_relative_to(workspace.resolve()):
        raise PermissionError(f"Access denied: {p} is outside workspace {workspace}")
    return p


class ReadFileTool(Tool):
    def __init__(self, workspace: Path, restrict: bool = True) -> None:
        self._workspace = workspace
        self._restrict = restrict

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read the contents of a file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "offset": {"type": "integer", "description": "Start line (0-based)", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to read", "default": 500},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path = _resolve_path(kwargs["path"], self._workspace, self._restrict)
        if not path.exists():
            return f"File not found: {path}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 500)
        selected = lines[offset : offset + limit]
        numbered = [f"{i + offset + 1:4d} | {line}" for i, line in enumerate(selected)]
        header = f"[{path.name}] lines {offset + 1}-{offset + len(selected)} of {len(lines)}"
        return header + "\n" + "\n".join(numbered)


class WriteFileTool(Tool):
    def __init__(self, workspace: Path, restrict: bool = True) -> None:
        self._workspace = workspace
        self._restrict = restrict

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories automatically."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path = _resolve_path(kwargs["path"], self._workspace, self._restrict)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(kwargs["content"], encoding="utf-8")
        return f"Written {len(kwargs['content'])} chars to {path}"


class EditFileTool(Tool):
    def __init__(self, workspace: Path, restrict: bool = True) -> None:
        self._workspace = workspace
        self._restrict = restrict

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace exact text in a file. old_text must match uniquely."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_text": {"type": "string", "description": "Exact text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path = _resolve_path(kwargs["path"], self._workspace, self._restrict)
        if not path.exists():
            return f"File not found: {path}"
        content = path.read_text(encoding="utf-8")
        old = kwargs["old_text"]
        count = content.count(old)
        if count == 0:
            return "old_text not found in file."
        if count > 1:
            return f"old_text matches {count} times. Make it more specific."
        content = content.replace(old, kwargs["new_text"], 1)
        path.write_text(content, encoding="utf-8")
        return f"Edited {path}"


class ListDirTool(Tool):
    def __init__(self, workspace: Path, restrict: bool = True) -> None:
        self._workspace = workspace
        self._restrict = restrict

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files and directories in a path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        path = _resolve_path(kwargs.get("path", "."), self._workspace, self._restrict)
        if not path.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:200]:
            prefix = "d " if entry.is_dir() else "f "
            lines.append(f"{prefix}{entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"
