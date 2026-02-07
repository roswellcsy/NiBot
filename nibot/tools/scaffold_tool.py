"""Scaffold tool -- generate project boilerplate from templates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nibot.registry import Tool
from nibot.types import ToolContext


# Built-in templates as dicts: {relative_path: content}
TEMPLATES: dict[str, dict[str, str]] = {
    "python-lib": {
        "pyproject.toml": '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{{name}}"
version = "0.1.0"
description = "{{description}}"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]
''',
        "{{name}}/__init__.py": '"""{{name}} -- {{description}}."""\n__version__ = "0.1.0"\n',
        "{{name}}/main.py": '"""Main module."""\n\n\ndef main() -> None:\n    print("Hello from {{name}}")\n\n\nif __name__ == "__main__":\n    main()\n',
        "tests/__init__.py": "",
        "tests/test_main.py": '"""Basic tests."""\nfrom {{name}}.main import main\n\ndef test_main():\n    main()  # should not raise\n',
        "README.md": "# {{name}}\n\n{{description}}\n",
        ".gitignore": "__pycache__/\n*.pyc\n.venv/\ndist/\n*.egg-info/\n",
    },
    "fastapi-app": {
        "pyproject.toml": '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{{name}}"
version = "0.1.0"
description = "{{description}}"
requires-python = ">=3.11"
dependencies = ["fastapi>=0.100", "uvicorn>=0.23"]
''',
        "{{name}}/__init__.py": "",
        "{{name}}/app.py": '''"""FastAPI application."""
from fastapi import FastAPI

app = FastAPI(title="{{name}}")

@app.get("/")
async def root():
    return {"message": "Hello from {{name}}"}

@app.get("/health")
async def health():
    return {"status": "ok"}
''',
        "{{name}}/main.py": '''"""Entry point."""
import uvicorn

def main():
    uvicorn.run("{{name}}.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
''',
        "tests/__init__.py": "",
        "tests/test_app.py": '''"""API tests."""
from fastapi.testclient import TestClient
from {{name}}.app import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200

def test_health():
    r = client.get("/health")
    assert r.json()["status"] == "ok"
''',
        "README.md": "# {{name}}\n\n{{description}}\n\n## Run\n\n```bash\npython -m {{name}}.main\n```\n",
        ".gitignore": "__pycache__/\n*.pyc\n.venv/\n",
    },
    "nibot-skill": {
        "SKILL.md": '''---
name: {{name}}
description: {{description}}
version: 1
---

## Instructions

[Describe when and how this skill should be used]

## Context

[Add relevant context, examples, or rules]
''',
        "README.md": "# {{name}}\n\nA NiBot skill: {{description}}\n",
    },
}


class ScaffoldTool(Tool):
    """Generate project boilerplate from built-in templates."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "scaffold"

    @property
    def description(self) -> str:
        templates = ", ".join(TEMPLATES.keys())
        return f"Generate project boilerplate. Templates: {templates}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "list"]},
                "template": {"type": "string", "description": "Template name"},
                "project_name": {"type": "string", "description": "Project name (lowercase, no spaces)"},
                "project_description": {"type": "string", "description": "Short project description"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]

        if action == "list":
            lines = []
            for name, files in TEMPLATES.items():
                lines.append(f"  {name}: {len(files)} files")
            return "Available templates:\n" + "\n".join(lines)

        if action == "create":
            template_name = kwargs.get("template", "")
            project_name = kwargs.get("project_name", "")
            description = kwargs.get("project_description", "")

            if not template_name:
                return "Error: 'template' is required."
            if not project_name:
                return "Error: 'project_name' is required."
            if template_name not in TEMPLATES:
                return f"Unknown template '{template_name}'. Available: {', '.join(TEMPLATES.keys())}"

            project_dir = self._workspace / project_name
            if project_dir.exists():
                return f"Error: directory '{project_name}' already exists."

            template = TEMPLATES[template_name]
            created_files = []
            for rel_path, content in template.items():
                # Variable substitution
                actual_path = rel_path.replace("{{name}}", project_name)
                actual_content = content.replace("{{name}}", project_name).replace(
                    "{{description}}", description or project_name
                )

                file_path = project_dir / actual_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(actual_content, encoding="utf-8")
                created_files.append(actual_path)

            return f"Project '{project_name}' created from '{template_name}':\n" + "\n".join(f"  {f}" for f in created_files)

        return f"Unknown action: {action}"
