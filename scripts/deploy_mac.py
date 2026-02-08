#!/usr/bin/env python3
"""NiBot deployment helper for Mac Mini M4 (Apple Silicon).

Checks environment, installs Ollama if needed, pulls recommended models,
and generates a hybrid config (local Ollama + cloud API).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check_platform() -> bool:
    """Verify macOS + Apple Silicon."""
    if platform.system() != "Darwin":
        print("Warning: This script is designed for macOS. Proceeding anyway...")
        return True
    machine = platform.machine()
    if machine != "arm64":
        print(f"Warning: Expected Apple Silicon (arm64), got {machine}")
    return True


def check_ollama() -> bool:
    """Check if Ollama is installed."""
    return shutil.which("ollama") is not None


def install_ollama() -> bool:
    """Attempt to install Ollama via brew."""
    print("Installing Ollama via Homebrew...")
    try:
        subprocess.run(["brew", "install", "ollama"], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Homebrew not available. Please install Ollama manually:")
        print("  https://ollama.ai/download")
        return False


def pull_models(models: list[str]) -> None:
    """Pull recommended models."""
    for model in models:
        print(f"Pulling {model}...")
        try:
            subprocess.run(["ollama", "pull", model], check=True, timeout=600)
            print(f"  {model} ready")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  Warning: Failed to pull {model}: {e}")


def generate_config(
    workspace: str = "~/.nibot/workspace",
    cloud_provider: str = "",
    cloud_api_key: str = "",
    local_model: str = "deepseek-r1:8b",
) -> dict:
    """Generate hybrid config: local Ollama for coding, cloud for main agent."""
    config = {
        "agent": {
            "model": f"ollama/{local_model}" if not cloud_provider else cloud_provider,
            "workspace": workspace,
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "providers": {
            "ollama": {
                "api_key": "ollama",
                "api_base": "http://localhost:11434",
            },
        },
        "agents": {
            "coder": {
                "model": f"ollama/{local_model}",
                "provider": "ollama",
                "tools": ["file_read", "write_file", "edit_file", "list_dir", "exec", "git",
                          "code_review", "test_runner"],
                "max_iterations": 25,
                "workspace_mode": "worktree",
            },
            "researcher": {
                "tools": ["web_search", "web_fetch", "file_read", "write_file"],
                "max_iterations": 15,
            },
        },
        "health": {"enabled": True, "port": 9100},
        "log": {"level": "INFO", "json_format": True, "file": "~/.nibot/nibot.log"},
    }

    if cloud_api_key:
        if "anthropic" in (cloud_provider or "").lower() or "claude" in (cloud_provider or "").lower():
            config["providers"]["anthropic"] = {"api_key": cloud_api_key}
            config["agent"]["model"] = cloud_provider or "anthropic/claude-sonnet-4-5-20250929"
        elif "openai" in (cloud_provider or "").lower() or "gpt" in (cloud_provider or "").lower():
            config["providers"]["openai"] = {"api_key": cloud_api_key}
            config["agent"]["model"] = cloud_provider or "gpt-4o"
        else:
            config["providers"]["openrouter"] = {"api_key": cloud_api_key}
            config["agent"]["model"] = cloud_provider or "openrouter/auto"

    return config


def generate_compose_override() -> dict:
    """Generate docker-compose.override.yml for Mac deployment."""
    return {
        "version": "3.8",
        "services": {
            "nibot": {
                "extra_hosts": ["host.docker.internal:host-gateway"],
                "environment": {
                    "NIBOT_PROVIDERS__OLLAMA__API_BASE": "http://host.docker.internal:11434",
                },
            },
        },
    }


def main() -> None:
    print("=== NiBot Mac Mini M4 Deployment ===\n")

    check_platform()

    # Ollama
    if not check_ollama():
        print("\nOllama not found.")
        if input("Install via Homebrew? [y/N] ").lower() == "y":
            if not install_ollama():
                sys.exit(1)
        else:
            print("Skipping Ollama installation.")

    # Models
    recommended = ["deepseek-r1:8b", "codellama:7b"]
    if check_ollama():
        print(f"\nRecommended models: {', '.join(recommended)}")
        if input("Pull recommended models? [y/N] ").lower() == "y":
            pull_models(recommended)

    # Config
    print("\n--- Configuration ---")
    cloud_key = input("Cloud API key (leave empty for local-only): ").strip()
    cloud_provider = ""
    if cloud_key:
        cloud_provider = input("Cloud provider model (e.g. anthropic/claude-sonnet-4-5-20250929): ").strip()

    config = generate_config(
        cloud_provider=cloud_provider,
        cloud_api_key=cloud_key,
    )

    config_path = Path.home() / ".nibot" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nConfig written to {config_path}")

    # Docker compose override
    override_path = Path("docker-compose.override.yml")
    try:
        override_path.write_text(
            json.dumps(generate_compose_override(), indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    print("\n=== Deployment ready! ===")
    print("Start with: docker-compose up -d")
    print("Or directly: python -m nibot")


if __name__ == "__main__":
    main()
