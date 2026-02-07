"""Logging configuration."""

from __future__ import annotations

import sys

from loguru import logger

# Remove loguru defaults; add a fallback sink so logging works before configure().
logger.remove()
_fallback_id: int | None = logger.add(
    sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}"
)


def configure(
    level: str = "INFO",
    fmt: str = "",
    json_format: bool = False,
    file: str = "",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Reconfigure logging sinks. Called once at startup from NiBot.__init__.

    Idempotent: removes all previous sinks and adds fresh ones.
    Before this is called, the fallback sink ensures basic stderr logging.
    """
    global _fallback_id
    logger.remove()  # remove ALL sinks including fallback
    _fallback_id = None

    if json_format:
        logger.add(sys.stderr, level=level, serialize=True)
    else:
        effective_fmt = fmt or "{time:HH:mm:ss} | {level:<7} | {message}"
        logger.add(sys.stderr, level=level, format=effective_fmt)

    if file:
        kw: dict = {"level": level, "rotation": rotation, "retention": retention}
        if json_format:
            kw["serialize"] = True
        else:
            kw["format"] = fmt or "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}"
        logger.add(file, **kw)
