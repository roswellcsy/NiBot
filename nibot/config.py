"""Configuration schema and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentTypeConfig(BaseModel):
    """Sub-agent type definition: tools whitelist, model override, system prompt."""

    tools: list[str] = Field(default_factory=list)
    model: str = ""
    max_iterations: int = 15
    system_prompt: str = ""
    provider: str = ""
    workspace_mode: str = ""


class ScheduledJob(BaseModel):
    """Cron-triggered task definition."""

    id: str = ""
    cron: str = ""
    prompt: str = ""
    channel: str = "scheduler"
    chat_id: str = ""
    enabled: bool = True


# Model name prefix -> provider config name mapping.
# Used by app._resolve_provider_credentials() and provider._configure_env_key().
MODEL_PROVIDER_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "gemini": "gemini",
    "moonshot": "kimi",
    "minimax": "minimax",
    "zhipu": "glm",
}

DEFAULT_AGENT_TYPES: dict[str, AgentTypeConfig] = {
    "coder": AgentTypeConfig(
        tools=["read_file", "write_file", "edit_file", "list_dir", "exec", "git",
               "code_review", "test_runner"],
        max_iterations=25,
        workspace_mode="worktree",
        system_prompt=(
            "You are a coding agent. Work in your isolated git worktree. "
            "Read existing code first, make minimal changes, test before declaring done."
        ),
    ),
    "researcher": AgentTypeConfig(
        tools=["web_search", "web_fetch", "read_file", "write_file"],
        max_iterations=15,
    ),
    "system": AgentTypeConfig(
        tools=["exec", "read_file", "list_dir"],
        max_iterations=10,
    ),
    "evolution": AgentTypeConfig(
        tools=["read_file", "write_file", "edit_file", "list_dir", "exec", "skill", "analyze"],
        max_iterations=30,
        system_prompt=(
            "You are NiBot's evolution engine. Your system state is injected above.\n\n"
            "WORKFLOW:\n"
            "1. Review metrics and skill inventory in your context.\n"
            "2. Use analyze(action='errors') to understand failure patterns.\n"
            "3. Use analyze(action='skill_impact', skill_name=X) to evaluate existing skills.\n"
            "4. Decide: create, update, disable, or skip.\n"
            "5. Execute using skill(action=create/update/disable/delete).\n"
            "6. Log decision: analyze(action='log_decision', ...).\n\n"
            "PRINCIPLES:\n"
            "- One improvement per run. Do not batch.\n"
            "- Disable before delete. Give skills a chance.\n"
            "- Skip if metrics are healthy.\n"
            "- Always log reasoning, even for 'skip'."
        ),
    ),
}


class AgentConfig(BaseModel):
    name: str = "NiBot"
    model: str = "anthropic/claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_iterations: int = 20
    workspace: str = "~/.nibot/workspace"
    bootstrap_files: list[str] = Field(
        default=["IDENTITY.md", "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    )
    context_window: int = 128000
    context_reserve: int = 4096
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    bus_queue_maxsize: int = 0
    gateway_tools: list[str] = Field(default_factory=list)
    auto_evolution: bool = False
    provider_fallback_chain: list[str] = Field(default_factory=list)


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
    wecom: "WeComChannelConfig" = Field(default_factory=lambda: WeComChannelConfig())
    api: "APIChannelConfig" = Field(default_factory=lambda: APIChannelConfig())


class ProviderConfig(BaseModel):
    api_key: str = ""
    api_base: str = ""
    model: str = ""


class ProvidersConfig(BaseModel):
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    extras: dict[str, ProviderConfig] = Field(default_factory=dict)

    def get(self, name: str) -> ProviderConfig | None:
        """Lookup by name: checks builtin fields first, then extras."""
        if hasattr(self, name) and name != "extras":
            val = getattr(self, name)
            if isinstance(val, ProviderConfig):
                return val
        return self.extras.get(name)


class ToolsConfig(BaseModel):
    restrict_to_workspace: bool = True
    exec_timeout: int = 60
    web_search_api_key: str = ""
    image_model: str = ""
    mcp_servers: dict[str, "MCPServerConfig"] = Field(default_factory=dict)
    pipeline_max_parallel: int = 5


class MCPServerConfig(BaseModel):
    """External MCP server to bridge into NiBot tool registry."""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration (sliding window)."""
    per_user_rpm: int = 30
    per_channel_rpm: int = 100
    enabled: bool = False


class WeComChannelConfig(BaseModel):
    enabled: bool = False
    corp_id: str = ""
    secret: str = ""
    agent_id: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    allow_from: list[str] = Field(default_factory=list)


class APIChannelConfig(BaseModel):
    enabled: bool = False
    auth_tokens: list[str] = Field(default_factory=list)


class WebhookServerConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080


class MarketplaceConfig(BaseModel):
    enabled: bool = False
    github_token: str = ""


class WebPanelConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9200
    auth_token: str = ""


class LogConfig(BaseModel):
    level: str = "INFO"
    format: str = "{time:HH:mm:ss} | {level:<7} | {message}"
    json_format: bool = False
    file: str = ""           # empty = no file output
    rotation: str = "10 MB"  # loguru rotation param
    retention: str = "7 days"


class HealthConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9100


class NiBotConfig(BaseSettings):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    webhook: WebhookServerConfig = Field(default_factory=WebhookServerConfig)
    web_panel: WebPanelConfig = Field(default_factory=WebPanelConfig)
    marketplace: MarketplaceConfig = Field(default_factory=MarketplaceConfig)
    agents: dict[str, AgentTypeConfig] = Field(default_factory=dict)
    schedules: list[ScheduledJob] = Field(default_factory=list)

    model_config = SettingsConfigDict(
        env_prefix="NIBOT_",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # env > dotenv > file (init) > defaults -- environment variables always win
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


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

    file_data: dict[str, Any] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            file_data = _convert_keys(raw)
        except (json.JSONDecodeError, ValueError):
            pass

    # Pass file data as kwargs so BaseSettings still applies env var overrides
    return NiBotConfig(**file_data)


def default_evolution_schedule() -> ScheduledJob:
    """Pre-configured daily evolution job (disabled by default)."""
    return ScheduledJob(
        id="evolution-daily",
        cron="0 3 * * *",
        prompt=(
            "Run a daily evolution cycle. "
            "Use analyze(action='metrics') to check system health. "
            "Use analyze(action='errors') to find patterns. "
            "Check skill(action='list') for current inventory. "
            "Create, update, or disable skills as needed. "
            "Log your decision: analyze(action='log_decision', ...)."
        ),
        channel="scheduler",
        enabled=False,
    )


def _default_config_path() -> Path:
    return Path.home() / ".nibot" / "config.json"


def validate_startup(config: NiBotConfig) -> None:
    """Validate config for production startup. Raises ValueError with all errors.

    Separated from NiBotConfig model_validator intentionally:
    tests construct NiBotConfig() without provider keys. Business rules
    (deployment constraints) != data structure constraints (pydantic).
    """
    errors: list[str] = []

    # 1. At least one provider must have an api_key or api_base (local providers like Ollama)
    builtins = [
        config.providers.anthropic, config.providers.openai,
        config.providers.openrouter, config.providers.deepseek,
    ]
    has_provider = any(p.api_key or p.api_base for p in builtins) or any(
        p.api_key or p.api_base for p in config.providers.extras.values()
    )
    if not has_provider:
        errors.append(
            "No provider configured. "
            "Set at least one api_key or api_base in providers.{name} or NIBOT_PROVIDERS__*__API_KEY"
        )

    # 2. Enabled channels must have credentials
    if config.channels.telegram.enabled and not config.channels.telegram.token:
        errors.append("channels.telegram.enabled=true but token is empty")
    if config.channels.feishu.enabled:
        if not config.channels.feishu.app_id or not config.channels.feishu.app_secret:
            errors.append("channels.feishu.enabled=true but app_id/app_secret missing")
    if config.channels.wecom.enabled:
        if not config.channels.wecom.corp_id or not config.channels.wecom.secret:
            errors.append("channels.wecom.enabled=true but corp_id/secret missing")

    # 3. Cron expressions must parse
    from croniter import croniter
    for job in config.schedules:
        if job.cron and not croniter.is_valid(job.cron):
            errors.append(f"schedule '{job.id}': invalid cron '{job.cron}'")

    # 4. Log level must be valid
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    if config.log.level.upper() not in valid_levels:
        errors.append(f"log.level '{config.log.level}' invalid, must be one of {valid_levels}")

    if errors:
        raise ValueError(
            "NiBot configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
