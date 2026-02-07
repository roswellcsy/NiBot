"""Git worktree manager -- isolated branch workspaces for coding agents."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nibot.log import logger


class WorktreeManager:
    """Manage git worktrees for isolated coding tasks."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._worktrees_dir = workspace / ".worktrees"

    async def _git(self, *args: str, cwd: Path | None = None) -> tuple[int, str, str]:
        """Run a git command, return (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd or self._workspace),
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode(), stderr.decode()

    async def ensure_repo(self) -> bool:
        """Init git repo in workspace if not already a repo. Returns True if repo exists/created."""
        rc, _, _ = await self._git("rev-parse", "--git-dir")
        if rc == 0:
            # Always ensure local git identity for commit operations
            await self._git("config", "user.email", "nibot@local")
            await self._git("config", "user.name", "NiBot")
            # Verify HEAD exists (has at least one commit)
            rc2, _, _ = await self._git("rev-parse", "HEAD")
            if rc2 == 0:
                return True
            # Repo exists but no commits -- fall through to create initial commit
        else:
            rc, _, err = await self._git("init")
            if rc != 0:
                logger.error(f"git init failed: {err}")
                return False
            await self._git("config", "user.email", "nibot@local")
            await self._git("config", "user.name", "NiBot")
        # Create initial commit so worktrees have a base
        rc, _, _ = await self._git("commit", "--allow-empty", "-m", "init")
        return rc == 0

    async def create(self, task_id: str, base_branch: str = "main") -> Path:
        """Create worktree at .worktrees/{task_id} on branch task/{task_id}."""
        self._worktrees_dir.mkdir(parents=True, exist_ok=True)
        wt_path = self._worktrees_dir / task_id
        branch = f"task/{task_id}"

        # Determine base: use base_branch if it exists, else HEAD
        rc, _, _ = await self._git("rev-parse", "--verify", base_branch)
        base = base_branch if rc == 0 else "HEAD"

        rc, out, err = await self._git(
            "worktree", "add", "-b", branch, str(wt_path), base,
        )
        if rc != 0:
            logger.error(f"worktree create failed: {err}")
            raise RuntimeError(f"git worktree add failed: {err.strip()}")
        return wt_path

    async def remove(self, task_id: str) -> bool:
        """Remove worktree and prune. Returns True if removed."""
        wt_path = self._worktrees_dir / task_id
        if not wt_path.exists():
            return False
        rc, _, err = await self._git("worktree", "remove", str(wt_path), "--force")
        if rc != 0:
            logger.warning(f"worktree remove failed: {err}")
            return False
        await self._git("worktree", "prune")
        return True

    async def commit(self, task_id: str, message: str) -> str:
        """Stage all changes and commit in worktree. Returns commit output."""
        wt_path = self._worktrees_dir / task_id
        await self._git("add", "-A", cwd=wt_path)
        rc, out, err = await self._git("commit", "-m", message, cwd=wt_path)
        if rc != 0:
            return err.strip() or "nothing to commit"
        return out.strip()

    async def diff(self, task_id: str) -> str:
        """Get diff stat of worktree vs HEAD (includes untracked files). Zero side effects."""
        wt_path = self._worktrees_dir / task_id
        # Tracked changes (no index mutation)
        _, tracked, _ = await self._git("diff", "--stat", "HEAD", cwd=wt_path)
        # Untracked files
        _, untracked, _ = await self._git(
            "ls-files", "--others", "--exclude-standard", cwd=wt_path,
        )
        parts = []
        if tracked.strip():
            parts.append(tracked.strip())
        if untracked.strip():
            files = untracked.strip().splitlines()
            parts.append(f"{len(files)} untracked file(s): " + ", ".join(files[:10]))
            if len(files) > 10:
                parts[-1] += f" ... (+{len(files) - 10} more)"
        return "\n".join(parts)

    async def status(self, task_id: str) -> str:
        """Get git status of worktree."""
        wt_path = self._worktrees_dir / task_id
        _, out, _ = await self._git("status", "--short", cwd=wt_path)
        return out.strip()

    async def log(self, task_id: str, n: int = 5) -> str:
        """Get recent commit log of worktree branch."""
        wt_path = self._worktrees_dir / task_id
        _, out, _ = await self._git("log", "--oneline", f"-{n}", cwd=wt_path)
        return out.strip()

    async def merge(self, task_id: str, base_branch: str = "main") -> str:
        """Merge task branch into base branch. Returns merge output or error."""
        branch = f"task/{task_id}"
        # Determine base: use base_branch if it exists, else default branch
        rc, _, _ = await self._git("rev-parse", "--verify", base_branch)
        if rc != 0:
            rc2, out2, _ = await self._git("symbolic-ref", "--short", "HEAD")
            base_branch = out2.strip() if rc2 == 0 else "main"

        # Checkout the base branch before merging
        rc, _, err = await self._git("checkout", base_branch)
        if rc != 0:
            return f"Merge failed: cannot checkout {base_branch}: {err.strip()}"

        rc, out, err = await self._git("merge", branch, "--no-ff",
                                       "-m", f"Merge {branch} into {base_branch}")
        if rc != 0:
            await self._git("merge", "--abort")
            return f"Merge failed: {err.strip()}"
        return out.strip() or f"Merged {branch} into {base_branch}."

    async def branch_info(self, task_id: str) -> dict[str, str]:
        """Return branch name, commit count, and last commit message for a task."""
        branch = f"task/{task_id}"
        wt_path = self._worktrees_dir / task_id

        # Commit count on branch vs base
        _, count_out, _ = await self._git("rev-list", "--count", f"HEAD..{branch}")
        count = count_out.strip() or "0"

        # Last commit message
        _, log_out, _ = await self._git("log", "-1", "--format=%s", branch)
        last_msg = log_out.strip()

        return {
            "branch": branch,
            "commits": count,
            "last_message": last_msg,
            "path": str(wt_path),
        }

    async def list_worktrees(self) -> list[dict[str, str]]:
        """List all worktrees with path and branch info."""
        rc, out, _ = await self._git("worktree", "list", "--porcelain")
        if rc != 0:
            return []
        result: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in out.splitlines():
            if line.startswith("worktree "):
                if current:
                    result.append(current)
                current = {"path": line[9:]}
            elif line.startswith("branch "):
                current["branch"] = line[7:]
            elif line == "bare":
                current["bare"] = "true"
        if current:
            result.append(current)
        return result
