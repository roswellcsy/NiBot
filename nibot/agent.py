"""Agent loop -- LLM + Tool iteration until completion."""

from __future__ import annotations

import json
from typing import Any

from nibot.bus import MessageBus
from nibot.config import NiBotConfig
from nibot.context import ContextBuilder
from nibot.log import logger
from nibot.provider import LLMProvider
from nibot.registry import ToolRegistry
from nibot.session import SessionManager
from nibot.types import Envelope


class AgentLoop:
    """Consume inbound messages, run LLM+Tool loop, publish outbound responses."""

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        registry: ToolRegistry,
        sessions: SessionManager,
        context_builder: ContextBuilder,
        config: NiBotConfig,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.registry = registry
        self.sessions = sessions
        self.context_builder = context_builder
        self.max_iterations = config.agent.max_iterations
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            envelope = await self.bus.consume_inbound()
            try:
                response = await self._process(envelope)
                await self.bus.publish_outbound(response)
            except Exception as e:
                logger.error(f"Agent error [{envelope.channel}:{envelope.chat_id}]: {e}")

    async def _process(self, envelope: Envelope) -> Envelope:
        session_key = f"{envelope.channel}:{envelope.chat_id}"
        session = self.sessions.get_or_create(session_key)

        messages = self.context_builder.build(session=session, current=envelope)
        tool_defs = self.registry.get_definitions()
        final_content = ""

        for _ in range(self.max_iterations):
            response = await self.provider.chat(messages=messages, tools=tool_defs or None)

            if not response.has_tool_calls:
                final_content = response.content or ""
                break

            # Append assistant message with tool_calls
            tc_dicts: list[dict[str, Any]] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": tc_dicts,
            })

            # Execute each tool, append results
            for tc in response.tool_calls:
                result = await self.registry.execute(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result.content,
                })

        # Persist to session
        session.add_message("user", envelope.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return Envelope(
            channel=envelope.channel,
            chat_id=envelope.chat_id,
            sender_id="assistant",
            content=final_content,
        )

    def stop(self) -> None:
        self._running = False
