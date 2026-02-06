"""Telegram channel -- python-telegram-bot integration."""

from __future__ import annotations

from typing import Any

from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import TelegramChannelConfig
from nibot.log import logger
from nibot.types import Envelope


class TelegramChannel(BaseChannel):
    name = "telegram"

    def __init__(self, config: TelegramChannelConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self._app: Any = None

    async def start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import ApplicationBuilder, MessageHandler, filters
        except ImportError:
            logger.error("python-telegram-bot not installed. pip install 'nibot[telegram]'")
            return

        self._app = ApplicationBuilder().token(self.config.token).build()

        async def on_message(update: Update, context: Any) -> None:
            msg = update.effective_message
            if not msg or not msg.text:
                return
            sender = str(update.effective_user.id) if update.effective_user else "unknown"
            chat_id = str(msg.chat_id)
            await self._handle_incoming(sender, chat_id, msg.text)

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
        self._running = True
        logger.info("Telegram channel starting...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        self._running = False
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send(self, envelope: Envelope) -> None:
        if not self._app:
            return
        try:
            await self._app.bot.send_message(
                chat_id=int(envelope.chat_id),
                text=envelope.content,
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
