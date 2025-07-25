"""Stateful Claude CLI client with session management and slash commands."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ._errors import CLIConnectionError, SessionError
from ._internal.session_persistence import SessionData, SessionPersistence, SessionSummary, ConversationMessage
from ._internal.slash_commands import SlashCommandRouter, CommandResult, SlashCommand
from ._internal.transport.subprocess_stateful_cli import StatefulCLITransport
from .types import ClaudeCodeOptions, ResponseMessage, TokenUsage

logger = logging.getLogger(__name__)


class SessionNotFoundError(SessionError):
    """Session doesn't exist in storage."""
    pass


class SessionNotActiveError(SessionError):
    """No active session for operation."""
    pass


class SessionCorruptedError(SessionError):
    """Session data is corrupted or invalid."""
    pass


class InvalidCommandError(SessionError):
    """Slash command is not recognized."""
    pass


class CommandExecutionError(SessionError):
    """Slash command failed to execute."""
    pass


class SessionInfo:
    """Session information container."""
    
    def __init__(self, session_data: SessionData):
        self.session_id = session_data.session_id
        self.start_time = session_data.start_time
        self.last_activity = session_data.last_activity
        self.message_count = session_data.message_count
        self.total_tokens = None  # Could be calculated from usage data
        self.working_directory = session_data.working_directory
        self.options = session_data.options


class StatefulCLIClient:
    """Persistent Claude CLI client with session management."""
    
    def __init__(
        self,
        session_id: str | None = None,
        options: ClaudeCodeOptions | None = None,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
        enable_slash_commands: bool = True
    ):
        """
        Initialize stateful Claude client.
        
        Args:
            session_id: Unique session identifier. If None, generates new UUID.
            options: Claude Code configuration options
            storage_path: Directory for session files (default: ~/.claude_sdk/sessions/)
            auto_save: Automatically save session state after each interaction
            enable_slash_commands: Enable built-in slash command processing
        """
        self._session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self._options = options or ClaudeCodeOptions()
        self._storage_path = storage_path
        self._auto_save = auto_save
        self._enable_slash_commands = enable_slash_commands
        
        self._transport: StatefulCLITransport | None = None
        self._persistence = SessionPersistence(storage_path)
        self._command_router = SlashCommandRouter() if enable_slash_commands else None
        self._active = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.end_session()

    async def start_session(self) -> str:
        """
        Start persistent interactive session.
        
        Returns:
            str: Active session ID
            
        Raises:
            CLIConnectionError: If unable to start CLI process
            SessionError: If session initialization fails
        """
        if self._active:
            return self._session_id
        
        try:
            self._transport = StatefulCLITransport(
                session_id=self._session_id,
                options=self._options,
                storage_path=self._storage_path
            )
            
            await self._transport.start_session(resume=True)
            self._active = True
            
            logger.info(f"Started session {self._session_id}")
            return self._session_id
            
        except Exception as e:
            logger.error(f"Failed to start session {self._session_id}: {e}")
            raise SessionError(f"Failed to start session: {e}") from e

    async def resume_session(self, session_id: str) -> bool:
        """
        Resume existing session from local storage.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            bool: True if session resumed successfully
            
        Raises:
            SessionNotFoundError: If session doesn't exist
            SessionCorruptedError: If session data is invalid
        """
        # Check if session exists
        if not await self._persistence.session_exists(session_id):
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        # End current session if active
        if self._active:
            await self.end_session(save=True)
        
        # Update session ID and start new transport
        self._session_id = session_id
        
        try:
            self._transport = StatefulCLITransport(
                session_id=self._session_id,
                options=self._options,
                storage_path=self._storage_path
            )
            
            await self._transport.start_session(resume=True)
            self._active = True
            
            logger.info(f"Resumed session {self._session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            raise SessionCorruptedError(f"Failed to resume session: {e}") from e

    async def end_session(self, save: bool = True) -> None:
        """
        End current session and cleanup resources.
        
        Args:
            save: Whether to save session state before ending
        """
        if not self._active:
            return
        
        try:
            if self._transport:
                await self._transport.end_session(save=save)
                self._transport = None
            
            self._active = False
            logger.info(f"Ended session {self._session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session {self._session_id}: {e}")
            raise

    def get_session_id(self) -> str | None:
        """Get current session ID."""
        return self._session_id if self._active else None

    def is_session_active(self) -> bool:
        """Check if session is currently active."""
        return self._active and (self._transport is not None) and self._transport.is_session_active()

    async def get_session_info(self) -> SessionInfo:
        """
        Get current session information.
        
        Returns:
            SessionInfo: Session metadata and statistics
        """
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        session_data = self._transport.get_session_data()
        if not session_data:
            raise SessionError("Session data not available")
        
        return SessionInfo(session_data)

    async def send_message(
        self, 
        message: str,
        stream_response: bool = True
    ) -> AsyncIterator[ResponseMessage] | ResponseMessage:
        """
        Send message to Claude and get response.
        
        Args:
            message: User message to send
            stream_response: If True, returns async iterator for streaming
            
        Returns:
            AsyncIterator[ResponseMessage] if streaming, single ResponseMessage otherwise
            
        Raises:
            SessionNotActiveError: If no active session
            CLIConnectionError: If communication fails
        """
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        # Check if this is a slash command
        if self._enable_slash_commands and self._command_router and self._command_router.is_command(message):
            result = await self._command_router.execute_command(message, self)
            
            # Return command result as ResponseMessage
            response = ResponseMessage(
                content=result.result or result.message or "",
                role="system",
                timestamp=datetime.now(),
                message_type="command_result",
                session_id=self._session_id,
                usage=None
            )
            
            if stream_response:
                async def command_response_stream():
                    yield response
                return command_response_stream()
            else:
                return response
        
        # Send regular message to Claude
        try:
            await self._transport.send_user_message(message)
            
            if stream_response:
                return self._stream_responses()
            else:
                # Collect all responses
                responses = []
                async for response in self._stream_responses():
                    responses.append(response)
                return responses[-1] if responses else ResponseMessage(
                    content="", role="assistant", timestamp=datetime.now(),
                    message_type="text", session_id=self._session_id
                )
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise CLIConnectionError(f"Failed to send message: {e}") from e

    async def _stream_responses(self) -> AsyncIterator[ResponseMessage]:
        """Stream responses from the transport."""
        if not self._transport:
            return
        
        try:
            async for message in self._transport.receive_messages():
                response = ResponseMessage(
                    content=message.get("content", ""),
                    role=message.get("role", "assistant"),
                    timestamp=datetime.now(),
                    message_type=message.get("type", "text"),
                    session_id=self._session_id,
                    usage=TokenUsage(**message["usage"]) if message.get("usage") else None
                )
                
                # Auto-save after each response if enabled
                if self._auto_save:
                    await self._transport.save_session()
                
                yield response
                
        except Exception as e:
            logger.error(f"Error streaming responses: {e}")
            raise

    async def send_command(self, command: str) -> CommandResult:
        """
        Send slash command (e.g., /help, /history).
        
        Args:
            command: Slash command string
            
        Returns:
            CommandResult: Command execution result
            
        Raises:
            InvalidCommandError: If command is not recognized
            CommandExecutionError: If command fails
        """
        if not self._enable_slash_commands or not self._command_router:
            raise InvalidCommandError("Slash commands are disabled")
        
        if not self._command_router.is_command(command):
            raise InvalidCommandError(f"Invalid command format: {command}")
        
        try:
            result = await self._command_router.execute_command(command, self)
            if not result.success:
                raise CommandExecutionError(result.message or "Command failed")
            return result
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise CommandExecutionError(f"Command failed: {e}") from e

    async def save_session(self) -> None:
        """Manually save current session state to disk."""
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        await self._transport.save_session()

    async def load_session(self, session_id: str) -> SessionData:
        """
        Load session data from storage.
        
        Args:
            session_id: Session to load
            
        Returns:
            SessionData: Complete session information
        """
        session_data = await self._persistence.load_session(session_id)
        if not session_data:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session_data

    async def export_session(
        self, 
        session_id: str, 
        export_path: Path | str,
        format: str = "json"
    ) -> None:
        """
        Export session to file.
        
        Args:
            session_id: Session to export
            export_path: Output file path
            format: Export format ('json', 'markdown', 'txt')
        """
        await self._persistence.export_session(session_id, export_path, format)

    async def list_sessions(
        self, 
        limit: int = 50,
        order_by: str = "last_activity"
    ) -> list[SessionSummary]:
        """
        List available sessions.
        
        Args:
            limit: Maximum sessions to return
            order_by: Sort field ('start_time', 'last_activity', 'message_count')
            
        Returns:
            list[SessionSummary]: Session summaries
        """
        return await self._persistence.list_sessions(limit, order_by)

    async def get_conversation_history(
        self,
        limit: int | None = None,
        since: datetime | None = None
    ) -> list[ConversationMessage]:
        """
        Get conversation history for current session.
        
        Args:
            limit: Maximum messages to return
            since: Only messages after this timestamp
            
        Returns:
            list[ConversationMessage]: Historical messages
        """
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        session_data = self._transport.get_session_data()
        if not session_data:
            return []
        
        messages = session_data.conversation_history
        
        # Filter by timestamp if specified
        if since:
            messages = [msg for msg in messages if msg.timestamp > since]
        
        # Apply limit if specified
        if limit:
            messages = messages[-limit:]
        
        return messages

    async def search_conversation(
        self,
        query: str,
        session_id: str | None = None
    ) -> list[ConversationMessage]:
        """
        Search conversation history.
        
        Args:
            query: Search term
            session_id: Session to search (current if None)
            
        Returns:
            list[ConversationMessage]: Matching messages
        """
        target_session_id = session_id or self._session_id
        session_data = await self._persistence.load_session(target_session_id)
        
        if not session_data:
            return []
        
        query_lower = query.lower()
        return [
            msg for msg in session_data.conversation_history
            if query_lower in msg.content.lower()
        ]

    async def clear_conversation_history(self) -> None:
        """Clear conversation history for current session."""
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        session_data = self._transport.get_session_data()
        if session_data:
            session_data.conversation_history.clear()
            await self._transport.save_session()

    async def register_command(
        self,
        command: str,
        handler: SlashCommand,
        description: str = ""
    ) -> None:
        """
        Register custom slash command.
        
        Args:
            command: Command name (without /)
            handler: SlashCommand instance to handle command
            description: Help text for command
        """
        if not self._enable_slash_commands or not self._command_router:
            raise InvalidCommandError("Slash commands are disabled")
        
        self._command_router.register_command(handler)

    async def unregister_command(self, command: str) -> None:
        """Remove custom command registration."""
        if not self._enable_slash_commands or not self._command_router:
            return
        
        self._command_router.unregister_command(command)

    async def get_working_directory(self) -> str | None:
        """Get current working directory for session."""
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        session_data = self._transport.get_session_data()
        return session_data.working_directory if session_data else None

    async def set_working_directory(self, path: str | Path) -> None:
        """Set working directory for session."""
        if not self._active or not self._transport:
            raise SessionNotActiveError("No active session")
        
        session_data = self._transport.get_session_data()
        if session_data:
            session_data.working_directory = str(path)
            session_data.metadata["working_directory_updated"] = datetime.now().isoformat()
            await self._transport.save_session()

    async def get_session_options(self) -> ClaudeCodeOptions:
        """Get current session configuration options."""
        return self._options

    async def update_session_options(self, options: ClaudeCodeOptions) -> None:
        """Update session configuration options."""
        self._options = options
        
        if self._active and self._transport:
            session_data = self._transport.get_session_data()
            if session_data:
                session_data.options = options
                await self._transport.save_session()