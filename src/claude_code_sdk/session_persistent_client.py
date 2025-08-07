"""Session Persistent Client that wraps ClaudeSDKClient for automatic session persistence."""

from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ._internal.session_storage import SessionData, SimpleSessionPersistence
from .client import ClaudeSDKClient
from .types import ClaudeCodeOptions, Message, ResultMessage


class SessionPersistentClient:
    """
    A wrapper around ClaudeSDKClient that provides automatic session persistence.
    
    This client saves all conversation messages and metadata to files automatically,
    allowing you to resume conversations and inspect session history later.
    
    Key features:
    - Wraps ClaudeSDKClient for all Claude interactions
    - Automatically extracts session IDs from received messages
    - Saves conversation history to JSON files
    - Provides session management (list, delete, inspect)
    - Uses server-generated session IDs (no client-side UUID generation)
    - Handles server-side session ID changes gracefully while preserving conversation history
    
    Session ID Handling:
    The Claude server may change session IDs during a conversation. This client handles
    such changes correctly by:
    - Preserving all conversation history when session ID changes
    - Moving the session data to the new session ID
    - Cleaning up old session files automatically
    - Maintaining session continuity and start times
    
    Example:
        ```python
        async with SessionPersistentClient() as client:
            await client.query("Hello, remember my name is Alice")
            
            async for message in client.receive_response():
                print(message)
            
            # Session is automatically saved with current session ID
            session_id = client.get_current_session_id()
            print(f"Session saved as: {session_id}")
            
            # Even if server changes session ID, history is preserved
        ```
    """
    
    def __init__(
        self,
        options: ClaudeCodeOptions | None = None,
        storage_path: Path | str | None = None,
    ):
        """
        Initialize the session persistent client.
        
        Args:
            options: Claude Code options to pass to underlying client
            storage_path: Directory to store session files.
                         This path is passed to SimpleSessionPersistence which creates the directory
                         if it doesn't exist and stores session JSON files there.
        """
        self._client = ClaudeSDKClient(options)
        self._persistence = SimpleSessionPersistence(storage_path)
        self._current_session_id: str | None = None
        self._session_data: SessionData | None = None
    
    @property
    def client(self) -> ClaudeSDKClient:
        """Access to the underlying ClaudeSDKClient."""
        return self._client
    
    async def connect(
        self, prompt: str | AsyncIterable[dict[str, Any]] | None = None
    ) -> None:
        """Connect to Claude with a prompt or message stream."""
        await self._client.connect(prompt)
    
    async def query(
        self, prompt: str | AsyncIterable[dict[str, Any]], session_id: str = "default"
    ) -> None:
        """Send a new request in streaming mode."""
        await self._client.query(prompt, session_id)
    
    async def interrupt(self) -> None:
        """Send interrupt signal (only works with streaming mode)."""
        await self._client.interrupt()
    
    async def receive_messages(self) -> AsyncIterator[Message]:
        """
        Receive all messages from Claude with automatic session persistence.
        
        This method wraps the underlying ClaudeSDKClient.receive_messages() and
        automatically extracts session data for persistence.
        """
        async for message in self._client.receive_messages():
            # Extract session data from message for persistence
            await self._handle_message_persistence(message)
            yield message
    
    async def receive_response(self) -> AsyncIterator[Message]:
        """
        Receive messages from Claude until ResultMessage with automatic persistence.
        
        This method wraps the underlying ClaudeSDKClient.receive_response() and
        automatically extracts session data for persistence.
        """
        async for message in self._client.receive_response():
            # Extract session data from message for persistence
            await self._handle_message_persistence(message)
            yield message
    
    async def start_or_resume_session(self, session_id: str | None = None) -> None:
        """
        Start a new session or resume an existing one.
        
        Args:
            session_id: If provided, attempts to resume the session with this ID.
                       If None, starts a new session.
                       
        Note:
            This method configures the underlying ClaudeSDKClient to use --resume <session_id>
            when connecting to Claude CLI, which allows resuming server-side conversation state.
        """
        if session_id:
            # Load existing session data if available
            self._session_data = await self._persistence.load_session(session_id)
            if self._session_data:
                self._current_session_id = session_id
                
            # Configure the underlying client to resume the session
            if self._client.options is None:
                from .types import ClaudeCodeOptions
                self._client.options = ClaudeCodeOptions()
            self._client.options.resume = session_id
        else:
            # Starting new session - clear any resume option
            if self._client.options:
                self._client.options.resume = None
            self._current_session_id = None
            self._session_data = None

    async def disconnect(self) -> None:
        """Disconnect from Claude and finalize session persistence."""
        # Update final session metadata before disconnecting
        if self._session_data:
            self._session_data.last_activity = datetime.now()
            await self._persistence.save_session(self._session_data)
        
        await self._client.disconnect()
    
    async def __aenter__(self) -> "SessionPersistentClient":
        """Enter async context - automatically connects."""
        await self.connect()
        return self
    
    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> bool:
        """Exit async context - automatically disconnects and saves session."""
        await self.disconnect()
        return False
    
    # Session Management Methods
    
    def get_current_session_id(self) -> str | None:
        """Get the current session ID (server-generated)."""
        return self._current_session_id
    
    async def list_sessions(self) -> list[str]:
        """List all saved session IDs."""
        return await self._persistence.list_sessions()
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a saved session.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            bool: True if session was deleted, False if not found
        """
        return await self._persistence.delete_session(session_id)
    
    async def load_session(self, session_id: str) -> SessionData | None:
        """
        Load session data for inspection.
        
        Args:
            session_id: The session ID to load
            
        Returns:
            SessionData | None: Session data if found, None otherwise
        """
        return await self._persistence.load_session(session_id)
    
    # Private Methods
    
    async def _handle_message_persistence(self, message: Message) -> None:
        """
        Handle automatic message persistence based on received messages.
        
        This method handles server-side session ID changes correctly:
        - When a session_id is first received, a new session is created
        - When a session_id changes during an active session, the existing
          conversation history is preserved and moved to the new session_id
        - Old session files are cleaned up when session_id changes
        
        Args:
            message: Message received from ClaudeSDKClient
        """
        # Extract session ID from message metadata if available
        session_id = getattr(message, 'session_id', None)
        
        # For ResultMessages, check if they have session_id
        if isinstance(message, ResultMessage) and hasattr(message, 'session_id'):
            session_id = message.session_id
        
        # Handle session ID updates from server
        if session_id:
            if session_id != self._current_session_id:
                # Session ID changed - this could be:
                # 1. First time getting a session ID (self._current_session_id is None)
                # 2. Server updated the session ID for the same logical session
                
                old_session_id = self._current_session_id
                self._current_session_id = session_id
                
                if self._session_data is not None:
                    # We have existing session data, so this is a session ID update
                    # Update the session ID in the existing session data
                    self._session_data.session_id = session_id
                    self._session_data.last_activity = datetime.now()
                    
                    # Save the session under the new session ID
                    await self._persistence.save_session(self._session_data)
                    
                    # Clean up the old session file if it exists
                    if old_session_id:
                        await self._persistence.delete_session(old_session_id)
                        
                else:
                    # No existing session data - try to load session with new session_id first
                    # (in case this is a resumed session), otherwise create new one
                    self._session_data = await self._persistence.load_session(session_id)
                    if not self._session_data:
                        self._session_data = SessionData(
                            session_id=session_id,
                            start_time=datetime.now(),
                            last_activity=datetime.now(),
                            conversation_history=[],
                            working_directory=str(Path.cwd()),
                            options=self._client.options,
                        )
        
        # Add message directly to session (Message objects are already the right type)
        if self._session_data:
            self._session_data.add_message(message)
            self._session_data.last_activity = datetime.now()
            
            # Save session after each message
            await self._persistence.save_session(self._session_data)