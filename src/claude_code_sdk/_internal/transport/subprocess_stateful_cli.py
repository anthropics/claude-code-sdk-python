"""Stateful subprocess transport implementation for persistent Claude Code CLI sessions."""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .subprocess_cli import SubprocessCLITransport
from ..._errors import CLIConnectionError
from ...types import ClaudeCodeOptions
from ..session_persistence import SessionData, SessionPersistence, ConversationMessage

logger = logging.getLogger(__name__)


class StatefulCLITransport(SubprocessCLITransport):
    """Stateful transport that maintains persistent CLI process for interactive sessions."""

    def __init__(
        self,
        session_id: str,
        options: ClaudeCodeOptions,
        cli_path: str | None = None,
        storage_path: Path | str | None = None,
    ):
        self._session_id = session_id
        self._input_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._session_active = False
        self._persistence = SessionPersistence(storage_path)
        self._session_data: SessionData | None = None
        
        # Create async generator for continuous input
        super().__init__(
            prompt=self._create_session_stream(),
            options=options,
            cli_path=cli_path,
            close_stdin_after_prompt=False,  # Keep stdin open for persistent session
        )

    async def _create_session_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Create async generator that yields messages from input queue."""
        while self._session_active:
            try:
                # Wait for messages with timeout to allow for clean shutdown
                message = await asyncio.wait_for(self._input_queue.get(), timeout=1.0)
                yield message
            except asyncio.TimeoutError:
                # Continue loop to check if session is still active
                continue
            except Exception as e:
                logger.debug(f"Error in session stream: {e}")
                break

    async def start_session(self, resume: bool = True) -> None:
        """Start persistent session."""
        if self._session_active:
            return
            
        # Try to load existing session if resuming
        if resume:
            self._session_data = await self._persistence.load_session(self._session_id)
        
        # Create new session if not found or not resuming
        if not self._session_data:
            now = datetime.now()
            self._session_data = SessionData(
                session_id=self._session_id,
                start_time=now,
                last_activity=now,
                options=self._options,
                working_directory=str(self._options.cwd) if self._options.cwd else None
            )
            await self._persistence.save_session(self._session_data)
            
        self._session_active = True
        
        # Update options to include session continuity
        if not self._options.continue_conversation and resume:
            # Create new options with continue flag
            self._options = ClaudeCodeOptions(
                **self._options.__dict__,
                continue_conversation=True,
                resume=self._session_id
            )
        
        await self.connect()

    async def end_session(self, save: bool = True) -> None:
        """End persistent session."""
        self._session_active = False
        
        # Save session data if requested
        if save and self._session_data:
            await self._persistence.save_session(self._session_data)
        
        # Clear any pending messages
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        await self.disconnect()

    async def send_user_message(self, content: str, message_type: str = "user") -> None:
        """Send user message to the persistent session."""
        if not self._session_active:
            raise CLIConnectionError("Session not active")

        # Add to session history
        if self._session_data:
            user_message = ConversationMessage(
                role="user",
                content=content,
                timestamp=datetime.now(),
                message_type=message_type
            )
            self._session_data.add_message(user_message)

        message = {
            "type": message_type,
            "message": {
                "role": "user",
                "content": content
            },
            "parent_tool_use_id": None,
            "session_id": self._session_id,
        }
        
        await self._input_queue.put(message)

    async def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages from CLI and save to session history."""
        async for message in super().receive_messages():
            # Add assistant messages to session history
            if self._session_data and message.get("type") in ["message", "text"]:
                assistant_message = ConversationMessage(
                    role="assistant",
                    content=message.get("content", ""),
                    timestamp=datetime.now(),
                    message_type=message.get("type", "text"),
                    usage=message.get("usage")
                )
                self._session_data.add_message(assistant_message)
            
            yield message

    def get_session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    def is_session_active(self) -> bool:
        """Check if session is active."""
        return self._session_active and self.is_connected()

    def get_session_data(self) -> SessionData | None:
        """Get current session data."""
        return self._session_data

    async def save_session(self) -> None:
        """Manually save session data."""
        if self._session_data:
            await self._persistence.save_session(self._session_data)

    def _build_command(self) -> list[str]:
        """Build CLI command with session-specific arguments."""
        cmd = super()._build_command()
        
        # Ensure we're using streaming input format for persistent session
        if "--input-format" not in cmd:
            cmd.extend(["--input-format", "stream-json"])
            
        return cmd