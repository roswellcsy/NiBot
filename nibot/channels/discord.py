"""Discord channel -- discord.py integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import DiscordChannelConfig
from nibot.log import logger
from nibot.types import Envelope


_DC_MAX_LENGTH = 2000


class DiscordChannel(BaseChannel):
    name = "discord"

    def __init__(self, config: DiscordChannelConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self._client: Any = None
        self._stream_msgs: dict[int, int] = {}  # channel_id -> message_id for streaming edits

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            logger.error("discord.py not installed. pip install 'nibot[discord]'")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            logger.info(f"Discord connected as {self._client.user}")

        @self._client.event
        async def on_message(message: Any) -> None:
            if message.author == self._client.user or message.author.bot:
                return
            # Guild: only respond when @mentioned; DM: always respond
            if message.guild and not self._client.user.mentioned_in(message):
                return
            content = message.content
            # Strip @mention prefix (both <@id> and <@!id> formats)
            if self._client.user:
                uid = str(self._client.user.id)
                content = content.replace(f"<@!{uid}>", "").replace(f"<@{uid}>", "").strip()
            if not content:
                return
            await self._handle_incoming(
                str(message.author.id), str(message.channel.id), content,
            )

        self._running = True
        task = asyncio.create_task(self._client.start(self.config.token))
        task.add_done_callback(self._on_client_done)

    def _on_client_done(self, task: asyncio.Task[None]) -> None:
        """Log Discord client failures instead of swallowing them."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Discord client failed: {exc!r}")
            self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._client and not self._client.is_closed():
            await self._client.close()

    async def send(self, envelope: Envelope) -> None:
        if not self._client or not self._client.is_ready():
            return
        try:
            channel = self._client.get_channel(int(envelope.chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(envelope.chat_id))
            meta = envelope.metadata or {}

            # Streaming chunks: edit message in place
            if meta.get("streaming"):
                await self._handle_stream_chunk(channel, envelope)
                return

            # Send media first
            for media_path in (envelope.media or []):
                p = Path(media_path)
                if p.exists():
                    import discord as _dc
                    await channel.send(file=_dc.File(str(p)))

            # Then send text (2000-char chunks)
            text = envelope.content
            if text:
                for i in range(0, max(1, len(text)), _DC_MAX_LENGTH):
                    await channel.send(text[i:i + _DC_MAX_LENGTH])
        except Exception as e:
            logger.error(f"Discord send error: {e}")

    async def _handle_stream_chunk(self, channel: Any, envelope: Envelope) -> None:
        """Edit-in-place streaming: send first chunk, edit subsequent ones."""
        seq = (envelope.metadata or {}).get("stream_seq", 0)
        text = envelope.content or ""
        if not text:
            return
        ch_id = channel.id
        try:
            if seq == 0:
                msg = await channel.send(text[:_DC_MAX_LENGTH])
                self._stream_msgs[ch_id] = msg.id
            else:
                msg_id = self._stream_msgs.get(ch_id)
                if msg_id:
                    msg = await channel.fetch_message(msg_id)
                    await msg.edit(content=text[:_DC_MAX_LENGTH])
                else:
                    msg = await channel.send(text[:_DC_MAX_LENGTH])
                    self._stream_msgs[ch_id] = msg.id
        except Exception as e:
            logger.debug(f"Discord stream edit (seq={seq}): {e}")
        # Clean up tracking on final chunk
        if (envelope.metadata or {}).get("stream_done"):
            self._stream_msgs.pop(ch_id, None)
