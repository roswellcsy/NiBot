"""File tools tests -- path traversal, edge cases, CRUD operations."""
from __future__ import annotations

from pathlib import Path

import pytest

from nibot.tools.file_tools import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _resolve_path,
)


# ---------------------------------------------------------------------------
# _resolve_path  security boundary
# ---------------------------------------------------------------------------

class TestResolvePath:

    def test_relative_resolves_under_workspace(self, tmp_path: Path) -> None:
        p = _resolve_path("foo.txt", tmp_path, restrict=True)
        assert p == (tmp_path / "foo.txt").resolve()

    def test_blocks_dot_dot_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError, match="outside workspace"):
            _resolve_path("../../etc/passwd", tmp_path, restrict=True)

    def test_blocks_absolute_outside(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError, match="outside workspace"):
            _resolve_path("/etc/passwd", tmp_path, restrict=True)

    def test_allows_absolute_inside(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "file.txt"
        p = _resolve_path(str(target), tmp_path, restrict=True)
        assert p == target.resolve()

    def test_restrict_false_allows_outside(self, tmp_path: Path) -> None:
        # Use a path guaranteed to be outside tmp_path on any OS
        outside = Path(tmp_path.anchor) / "some_other_dir" / "file.txt"
        p = _resolve_path(str(outside), tmp_path, restrict=False)
        assert p == outside.resolve()

    def test_blocks_encoded_traversal(self, tmp_path: Path) -> None:
        """Path with embedded .. after expansion."""
        with pytest.raises(PermissionError):
            _resolve_path("subdir/../../secret", tmp_path, restrict=True)


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class TestReadFileTool:

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3", encoding="utf-8")
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(path="hello.txt")
        assert "line1" in result
        assert "line2" in result
        assert "[hello.txt]" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(path="nope.txt")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(path="big.txt", offset=10, limit=5)
        assert "line10" in result
        assert "line14" in result
        assert "line15" not in result

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        tool = ReadFileTool(workspace=tmp_path)
        result = await tool.execute(path="empty.txt")
        assert "[empty.txt]" in result

    @pytest.mark.asyncio
    async def test_read_path_traversal_blocked(self, tmp_path: Path) -> None:
        tool = ReadFileTool(workspace=tmp_path, restrict=True)
        with pytest.raises(PermissionError):
            await tool.execute(path="../../etc/passwd")


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------

class TestWriteFileTool:

    @pytest.mark.asyncio
    async def test_write_creates_file(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path)
        result = await tool.execute(path="new.txt", content="hello world")
        assert "Written" in result
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path)
        result = await tool.execute(path="a/b/c/deep.txt", content="deep")
        assert "Written" in result
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "exist.txt"
        f.write_text("old", encoding="utf-8")
        tool = WriteFileTool(workspace=tmp_path)
        await tool.execute(path="exist.txt", content="new")
        assert f.read_text(encoding="utf-8") == "new"

    @pytest.mark.asyncio
    async def test_write_path_traversal_blocked(self, tmp_path: Path) -> None:
        tool = WriteFileTool(workspace=tmp_path, restrict=True)
        with pytest.raises(PermissionError):
            await tool.execute(path="../../evil.txt", content="pwned")


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------

class TestEditFileTool:

    @pytest.mark.asyncio
    async def test_edit_unique_match(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n", encoding="utf-8")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(path="code.py", old_text="return 1", new_text="return 42")
        assert "Edited" in result
        assert "return 42" in f.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_edit_old_text_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    pass\n", encoding="utf-8")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(path="code.py", old_text="nonexistent", new_text="x")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_multiple_matches_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "dup.py"
        f.write_text("a = 1\nb = 1\n", encoding="utf-8")
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(path="dup.py", old_text="= 1", new_text="= 2")
        assert "matches 2 times" in result

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, tmp_path: Path) -> None:
        tool = EditFileTool(workspace=tmp_path)
        result = await tool.execute(path="nope.py", old_text="x", new_text="y")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------

class TestListDirTool:

    @pytest.mark.asyncio
    async def test_list_populated_dir(self, tmp_path: Path) -> None:
        (tmp_path / "afile.txt").touch()
        (tmp_path / "bdir").mkdir()
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(path=".")
        assert "f afile.txt" in result
        assert "d bdir" in result

    @pytest.mark.asyncio
    async def test_list_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(path="empty")
        assert "empty directory" in result.lower()

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(path="nope")
        assert "not a directory" in result.lower()

    @pytest.mark.asyncio
    async def test_list_dirs_sorted_first(self, tmp_path: Path) -> None:
        (tmp_path / "zfile").touch()
        (tmp_path / "adir").mkdir()
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(path=".")
        lines = result.strip().splitlines()
        assert lines[0].startswith("d ")  # dir first

    @pytest.mark.asyncio
    async def test_list_truncates_at_200(self, tmp_path: Path) -> None:
        for i in range(250):
            (tmp_path / f"file_{i:03d}.txt").touch()
        tool = ListDirTool(workspace=tmp_path)
        result = await tool.execute(path=".")
        lines = result.strip().splitlines()
        assert len(lines) == 200
