"""CLI entry point for NiBot."""

import asyncio
import argparse


def main():
    parser = argparse.ArgumentParser(description="NiBot AI Agent")
    parser.add_argument("--config", default=None, help="Path to config file")
    args = parser.parse_args()

    from nibot.app import NiBot

    app = NiBot(config_path=args.config)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
