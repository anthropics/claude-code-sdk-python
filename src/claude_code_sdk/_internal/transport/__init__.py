"""Transport implementations for Claude SDK."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from claude_code_sdk.types import ClaudeCodeOptions


class Transport(ABC):
    """Abstract transport for Claude communication."""

    @abstractmethod
    def configure(self, prompt: str, options: ClaudeCodeOptions) -> None:
        """Configure transport with prompt and options."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send_request(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> None:
        """Send request to Claude."""
        pass

    @abstractmethod
    def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages from Claude."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        pass


# Import implementations
__all__ = ["Transport"]
