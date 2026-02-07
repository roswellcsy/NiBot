"""Git tool -- worktree management, commit, diff, status, log."""

from __future__ import annotations

from typing import Any

from nibot.registry import Tool
from nibot.worktree import WorktreeManager


class GitTool(Tool):
    def __init__(self, worktree_mgr: WorktreeManager, allowed_task_id: str = "") -> None:
        self._wt = worktree_mgr
        self._allowed_task_id = allowed_task_id

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Git version control: worktree management, commit, diff, status, log. "
            "Use worktree_create to get an isolated branch for coding tasks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "worktree_create", "worktree_remove", "worktree_list",
                        "merge", "branch_info",
                        "commit", "diff", "status", "log",
                    ],
                    "description": "Git action to perform",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID for worktree operations",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (for commit action)",
                },
                "base_branch": {
                    "type": "string",
                    "description": "Base branch for worktree_create (default: main)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)
        action = kwargs["action"]
        task_id = kwargs.get("task_id", "")

        # Enforce isolation: if restricted, only allow operations on own task
        if self._allowed_task_id and task_id and task_id != self._allowed_task_id:
            return f"Access denied: you can only operate on task '{self._allowed_task_id}'"
        # If restricted and no task_id provided, use the allowed one
        if self._allowed_task_id and not task_id:
            task_id = self._allowed_task_id

        if action == "worktree_create":
            if not task_id:
                return "Error: task_id required for worktree_create"
            await self._wt.ensure_repo()
            base = kwargs.get("base_branch", "main")
            try:
                path = await self._wt.create(task_id, base)
                return f"Worktree created: {path} (branch: task/{task_id})"
            except RuntimeError as e:
                return f"Error: {e}"

        if action == "worktree_remove":
            if not task_id:
                return "Error: task_id required for worktree_remove"
            ok = await self._wt.remove(task_id)
            return f"Worktree removed: {task_id}" if ok else f"Worktree not found: {task_id}"

        if action == "worktree_list":
            wts = await self._wt.list_worktrees()
            if not wts:
                return "No worktrees found."
            lines = [f"  {w.get('branch', '?')} -> {w.get('path', '?')}" for w in wts]
            return "Worktrees:\n" + "\n".join(lines)

        if action == "merge":
            if not task_id:
                return "Error: task_id required for merge"
            base = kwargs.get("base_branch", "main")
            return await self._wt.merge(task_id, base)

        if action == "branch_info":
            if not task_id:
                return "Error: task_id required for branch_info"
            info = await self._wt.branch_info(task_id)
            return (f"Branch: {info['branch']}\n"
                    f"Commits: {info['commits']}\n"
                    f"Last: {info['last_message']}\n"
                    f"Path: {info['path']}")

        if action == "commit":
            if not task_id:
                return "Error: task_id required for commit"
            msg = kwargs.get("message", "auto-commit")
            return await self._wt.commit(task_id, msg)

        if action == "diff":
            if not task_id:
                return "Error: task_id required for diff"
            out = await self._wt.diff(task_id)
            return out or "No changes."

        if action == "status":
            if not task_id:
                return "Error: task_id required for status"
            out = await self._wt.status(task_id)
            return out or "Clean."

        if action == "log":
            if not task_id:
                return "Error: task_id required for log"
            out = await self._wt.log(task_id)
            return out or "No commits."

        return f"Unknown action: {action}"
