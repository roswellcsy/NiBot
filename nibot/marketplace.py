"""Skill marketplace -- discover and install skills from GitHub."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nibot.log import logger


@dataclass
class MarketplaceSkill:
    """A skill available in the marketplace."""
    name: str
    description: str
    url: str
    stars: int = 0
    author: str = ""
    version: str = ""


class SkillMarketplace:
    """Search and install skills from GitHub repositories.

    Skills are GitHub repos with topic 'nibot-skill' containing SKILL.md.
    """

    def __init__(self, github_token: str = "", skills_dir: Path | None = None) -> None:
        self._token = github_token
        self._skills_dir = skills_dir

    async def search(self, query: str = "", limit: int = 20) -> list[MarketplaceSkill]:
        """Search GitHub for nibot-skill repositories."""
        import httpx

        search_query = f"topic:nibot-skill {query}".strip()
        url = "https://api.github.com/search/repositories"
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    params={"q": search_query, "sort": "stars", "per_page": limit},
                    headers=headers,
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"Marketplace search failed: {e}")
            return []

        results = []
        for item in data.get("items", []):
            results.append(MarketplaceSkill(
                name=item.get("name", ""),
                description=item.get("description", "") or "",
                url=item.get("html_url", ""),
                stars=item.get("stargazers_count", 0),
                author=item.get("owner", {}).get("login", ""),
            ))
        return results

    async def install(self, url: str, name: str = "") -> str:
        """Install a skill from a GitHub repository URL.

        Downloads SKILL.md from the repo's default branch.
        """
        import httpx

        if not self._skills_dir:
            return "Error: no skills directory configured"

        # Parse GitHub URL to get owner/repo
        match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
        if not match:
            return f"Error: invalid GitHub URL: {url}"
        owner, repo = match.group(1), match.group(2)
        skill_name = name or repo

        # Download SKILL.md from main branch
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/SKILL.md"
        headers = {}
        if self._token:
            headers["Authorization"] = f"token {self._token}"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(raw_url, headers=headers, timeout=15.0)
                if resp.status_code == 404:
                    # Try master branch
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/SKILL.md"
                    resp = await client.get(raw_url, headers=headers, timeout=15.0)
                resp.raise_for_status()
                content = resp.text
        except Exception as e:
            return f"Error downloading SKILL.md: {e}"

        # Save to skills directory
        skill_dir = self._skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        (skill_dir / ".source").write_text(
            json.dumps({"url": url, "owner": owner, "repo": repo}, indent=2),
            encoding="utf-8",
        )

        return f"Skill '{skill_name}' installed from {url}"

    async def check_update(self, name: str) -> str:
        """Check if a skill has updates available."""
        if not self._skills_dir:
            return "Error: no skills directory configured"

        skill_dir = self._skills_dir / name
        source_file = skill_dir / ".source"
        if not source_file.exists():
            return f"Skill '{name}' has no source info (manually created?)"

        try:
            source = json.loads(source_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return f"Error reading source info for '{name}'"

        owner = source.get("owner", "")
        repo = source.get("repo", "")
        if not owner or not repo:
            return f"Invalid source info for '{name}'"

        # Compare local vs remote SKILL.md
        import httpx
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/SKILL.md"
        headers = {}
        if self._token:
            headers["Authorization"] = f"token {self._token}"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(raw_url, headers=headers, timeout=15.0)
                resp.raise_for_status()
                remote_content = resp.text
        except Exception as e:
            return f"Error checking update: {e}"

        local_content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        if local_content == remote_content:
            return f"Skill '{name}' is up to date."
        return f"Skill '{name}' has updates available. Use install to update."
