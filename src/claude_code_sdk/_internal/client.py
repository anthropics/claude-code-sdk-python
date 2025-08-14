"""Internal client implementation."""

from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any

from ..types import ClaudeCodeOptions, Message
from .message_parser import parse_message
from .transport.subprocess_cli import SubprocessCLITransport


class InternalClient:
    """Internal client implementation."""

    def __init__(self) -> None:
        """Initialize the internal client."""

    async def process_query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: ClaudeCodeOptions,
        cli_path: Path | None = None
    ) -> AsyncIterator[Message]:
        """Process a query through transport."""

        transport = SubprocessCLITransport(
            prompt=prompt, options=options, cli_path=cli_path, close_stdin_after_prompt=True
        )

        try:
            await transport.connect()

            async for data in transport.receive_messages():
                yield parse_message(data)

        finally:
            await transport.disconnect()
