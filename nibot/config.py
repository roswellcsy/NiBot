"""Configuration schema and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseModel):
    name: str = "NiBot"
    model: str = "anthropic/claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_iterations: int = 20
    workspace: str = "~/.nibot/workspace"


class TelegramChannelConfig(BaseModel):
    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)


class FeishuChannelConfig(BaseModel):
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    allow_from: list[str] = Field(default_factory=list)


class ChannelsConfig(BaseModel):
    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    feishu: FeishuChannelConfig = Field(default_factory=FeishuChannelConfig)


class ProviderConfig(BaseModel):
    api_key: str = ""
    api_base: str = ""


class ProvidersConfig(BaseModel):
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)


class ToolsConfig(BaseModel):
    restrict_to_workspace: bool = True
    exec_timeout: int = 60
    web_search_api_key: str = ""


class NiBotConfig(BaseSettings):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    model_config = SettingsConfigDict(
        env_prefix="NIBOT_",
        env_nested_delimiter="__",
    )


def _camel_to_snake(name: str) -> str:
    import re

    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _convert_keys(data: Any) -> Any:
    if isinstance(data, dict):
        return {_camel_to_snake(k): _convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_keys(i) for i in data]
    return data


def load_config(config_path: str | None = None) -> NiBotConfig:
    """Load config from JSON file + environment variables."""
    path = Path(config_path).expanduser() if config_path else _default_config_path()

    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            data = _convert_keys(raw)
            return NiBotConfig.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

    return NiBotConfig()


def _default_config_path() -> Path:
    return Path.home() / ".nibot" / "config.json"
