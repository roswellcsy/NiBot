"""Tests for v0.9.0b features: deployment readiness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nibot.config import (
    ChannelsConfig,
    FeishuChannelConfig,
    HealthConfig,
    LogConfig,
    NiBotConfig,
    ProviderConfig,
    ProvidersConfig,
    ScheduledJob,
    TelegramChannelConfig,
    validate_startup,
)


# ---- helpers ----

def _valid_config(**overrides: Any) -> NiBotConfig:
    """Create a NiBotConfig that passes validate_startup (has a provider key)."""
    defaults: dict[str, Any] = {
        "providers": ProvidersConfig(anthropic=ProviderConfig(api_key="sk-test")),
    }
    defaults.update(overrides)
    return NiBotConfig(**defaults)


# ---- B1: Config Startup Validation ----

class TestConfigValidation:
    def test_no_provider_key_raises(self) -> None:
        cfg = NiBotConfig()
        with pytest.raises(ValueError, match="No provider configured"):
            validate_startup(cfg)

    def test_one_provider_key_passes(self) -> None:
        cfg = _valid_config()
        validate_startup(cfg)  # should not raise

    def test_extras_provider_key_passes(self) -> None:
        cfg = NiBotConfig(
            providers=ProvidersConfig(
                extras={"custom": ProviderConfig(api_key="sk-custom")}
            )
        )
        validate_startup(cfg)  # should not raise

    def test_telegram_enabled_no_token_raises(self) -> None:
        cfg = _valid_config(
            channels=ChannelsConfig(
                telegram=TelegramChannelConfig(enabled=True, token="")
            ),
        )
        with pytest.raises(ValueError, match="telegram.*token"):
            validate_startup(cfg)

    def test_telegram_enabled_with_token_passes(self) -> None:
        cfg = _valid_config(
            channels=ChannelsConfig(
                telegram=TelegramChannelConfig(enabled=True, token="bot:tok")
            ),
        )
        validate_startup(cfg)

    def test_feishu_enabled_no_credentials_raises(self) -> None:
        cfg = _valid_config(
            channels=ChannelsConfig(
                feishu=FeishuChannelConfig(enabled=True, app_id="", app_secret="")
            ),
        )
        with pytest.raises(ValueError, match="feishu.*app_id"):
            validate_startup(cfg)

    def test_feishu_enabled_with_credentials_passes(self) -> None:
        cfg = _valid_config(
            channels=ChannelsConfig(
                feishu=FeishuChannelConfig(enabled=True, app_id="id", app_secret="sec")
            ),
        )
        validate_startup(cfg)

    def test_invalid_cron_raises(self) -> None:
        cfg = _valid_config(
            schedules=[ScheduledJob(id="bad", cron="not a cron")],
        )
        with pytest.raises(ValueError, match="invalid cron"):
            validate_startup(cfg)

    def test_valid_cron_passes(self) -> None:
        cfg = _valid_config(
            schedules=[ScheduledJob(id="good", cron="0 3 * * *")],
        )
        validate_startup(cfg)

    def test_invalid_log_level_raises(self) -> None:
        cfg = _valid_config(log=LogConfig(level="INVALID"))
        with pytest.raises(ValueError, match="log.level"):
            validate_startup(cfg)

    def test_valid_log_level_passes(self) -> None:
        cfg = _valid_config(log=LogConfig(level="DEBUG"))
        validate_startup(cfg)

    def test_multiple_errors_reported(self) -> None:
        """All errors should be collected, not just the first one."""
        cfg = NiBotConfig(
            channels=ChannelsConfig(
                telegram=TelegramChannelConfig(enabled=True, token="")
            ),
            log=LogConfig(level="BOGUS"),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_startup(cfg)
        msg = str(exc_info.value)
        assert "No provider configured" in msg
        assert "telegram" in msg
        assert "log.level" in msg

    def test_nibot_config_bare_construction_still_works(self) -> None:
        """Existing test pattern: NiBotConfig() must not raise."""
        cfg = NiBotConfig()
        assert cfg.agent.model == "anthropic/claude-opus-4-6"


# ---- B2: Health Check ----

class TestHealthServer:
    @pytest.mark.asyncio
    async def test_build_health_ok(self) -> None:
        from nibot.health import _build_health

        app = MagicMock()
        app.config.agent.model = "test-model"
        app.sessions._cache = {"k1": None, "k2": None}
        app._channels = [MagicMock(name="telegram")]
        app.agent._running = True
        app.agent._tasks = set()
        app.scheduler._jobs = {"j1": None}

        result = _build_health(app)
        assert result["status"] == "ok"
        assert result["model"] == "test-model"
        assert result["active_sessions"] == 2
        assert result["scheduler_jobs"] == 1

    @pytest.mark.asyncio
    async def test_build_health_degraded(self) -> None:
        from nibot.health import _build_health

        app = MagicMock()
        app.config.agent.model = "m"
        app.sessions._cache = {}
        app._channels = []
        app.agent._running = False
        app.agent._tasks = set()
        app.scheduler._jobs = {}

        result = _build_health(app)
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self) -> None:
        from nibot.health import start_health_server

        app = MagicMock()
        app.config.health.enabled = False
        server = await start_health_server(app)
        assert server is None

    @pytest.mark.asyncio
    async def test_server_starts_and_responds(self) -> None:
        """Integration: start real server, make HTTP request, verify JSON."""
        from nibot.health import start_health_server

        app = MagicMock()
        app.config.health.enabled = True
        app.config.health.host = "127.0.0.1"
        app.config.health.port = 0  # OS picks a free port
        app.config.agent.model = "test"
        app.sessions._cache = {}
        app._channels = []
        app.agent._running = True
        app.agent._tasks = set()
        app.scheduler._jobs = {}

        server = await start_health_server(app)
        assert server is not None
        try:
            addr = server.sockets[0].getsockname()
            port = addr[1]

            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()
            await writer.wait_closed()

            text = data.decode()
            assert "200 OK" in text
            body = text.split("\r\n\r\n", 1)[1]
            parsed = json.loads(body)
            assert parsed["status"] == "ok"
            assert parsed["model"] == "test"
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_404_for_unknown_path(self) -> None:
        from nibot.health import start_health_server

        app = MagicMock()
        app.config.health.enabled = True
        app.config.health.host = "127.0.0.1"
        app.config.health.port = 0
        app.config.agent.model = "m"
        app.sessions._cache = {}
        app._channels = []
        app.agent._running = True
        app.agent._tasks = set()
        app.scheduler._jobs = {}

        server = await start_health_server(app)
        assert server is not None
        try:
            addr = server.sockets[0].getsockname()
            port = addr[1]

            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()
            await writer.wait_closed()

            assert "404 Not Found" in data.decode()
        finally:
            server.close()
            await server.wait_closed()


# ---- B4: Structured Logging ----

class TestLogConfig:
    def test_default_log_config(self) -> None:
        cfg = NiBotConfig()
        assert cfg.log.level == "INFO"
        assert cfg.log.json_format is False
        assert cfg.log.file == ""

    def test_json_log_config(self) -> None:
        cfg = NiBotConfig(log=LogConfig(json_format=True, level="DEBUG"))
        assert cfg.log.json_format is True
        assert cfg.log.level == "DEBUG"

    def test_configure_runs_without_error(self) -> None:
        from nibot.log import configure
        configure(level="DEBUG", json_format=True)
        configure(level="INFO")  # reset

    def test_configure_with_file(self, tmp_path: Path) -> None:
        from nibot.log import configure, logger
        log_file = str(tmp_path / "test.log")
        configure(level="DEBUG", file=log_file)
        logger.info("test message for file output")
        assert (tmp_path / "test.log").exists()
        content = (tmp_path / "test.log").read_text(encoding="utf-8")
        assert "test message" in content
        configure(level="INFO")  # reset


# ---- Config data structure defaults ----

class TestHealthConfigDefaults:
    def test_health_disabled_by_default(self) -> None:
        cfg = NiBotConfig()
        assert cfg.health.enabled is False
        assert cfg.health.port == 9100

    def test_health_config_override(self) -> None:
        cfg = NiBotConfig(health=HealthConfig(enabled=True, port=8080))
        assert cfg.health.enabled is True
        assert cfg.health.port == 8080
