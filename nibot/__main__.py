"""CLI entry point for NiBot."""

import asyncio
import argparse


def main():
    parser = argparse.ArgumentParser(description="NiBot AI Agent")
    parser.add_argument("--config", default=None, help="Path to config file")
    args = parser.parse_args()

    from nibot.app import NiBot

    app = NiBot(config_path=args.config)
    _auto_add_channels(app)
    asyncio.run(app.run())


def _auto_add_channels(app):
    """Wire channels based on config."""
    cfg = app.config.channels
    if cfg.telegram.enabled and cfg.telegram.token:
        from nibot.channels.telegram import TelegramChannel

        app.add_channel(TelegramChannel(cfg.telegram, app.bus))
    if cfg.feishu.enabled and cfg.feishu.app_id:
        from nibot.channels.feishu import FeishuChannel

        app.add_channel(FeishuChannel(cfg.feishu, app.bus))


if __name__ == "__main__":
    main()
