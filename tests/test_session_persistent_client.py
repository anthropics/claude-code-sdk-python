"""
Unit tests for SessionPersistentClient.

This test suite covers:
- SessionData class: serialization, deserialization, message management  
- SimpleSessionPersistence class: file-based storage operations
- SessionPersistentClient class: wrapper functionality and automatic persistence

Key test scenarios:
- Automatic session creation when receiving messages with session_id
- Message persistence throughout conversation flow
- Session management operations (list, load, delete)
- Proper delegation to underlying ClaudeSDKClient
- Error handling for corrupted files and missing sessions
- Context manager behavior and cleanup
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    UserMessage,
)
from claude_code_sdk._internal.session_storage import SessionData, SimpleSessionPersistence
from claude_code_sdk.session_persistent_client import SessionPersistentClient


class TestSessionData:
    """Test SessionData class."""

    def test_session_data_initialization(self):
        """Test SessionData initialization."""
        session_id = "test-session-123"
        start_time = datetime.now()
        
        session_data = SessionData(
            session_id=session_id,
            start_time=start_time,
            last_activity=start_time,
            working_directory="/test/dir",
        )
        
        assert session_data.session_id == session_id
        assert session_data.start_time == start_time
        assert session_data.last_activity == start_time
        assert session_data.working_directory == "/test/dir"
        assert len(session_data.conversation_history) == 0

    def test_add_message(self):
        """Test adding messages to session data."""
        session_data = SessionData(
            session_id="test-session",
            start_time=datetime.now(),
            last_activity=datetime.now(),
        )
        
        # Add a user message
        user_msg = UserMessage(content="Hello")
        session_data.add_message(user_msg)
        
        assert len(session_data.conversation_history) == 1
        assert session_data.conversation_history[0] == user_msg
        
        # Add an assistant message
        assistant_msg = AssistantMessage(content=[TextBlock(text="Hi there!")])
        session_data.add_message(assistant_msg)
        
        assert len(session_data.conversation_history) == 2
        assert session_data.conversation_history[1] == assistant_msg

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        # Create session data with various message types
        start_time = datetime(2025, 1, 1, 12, 0, 0)
        last_activity = datetime(2025, 1, 1, 12, 5, 0)
        
        session_data = SessionData(
            session_id="test-session-456",
            start_time=start_time,
            last_activity=last_activity,
            working_directory="/test/path",
            options=ClaudeCodeOptions(model="claude-3-5-sonnet-20241022"),
        )
        
        # Add different message types (manually to control timing)
        session_data.conversation_history.append(UserMessage(content="Test user message"))
        session_data.conversation_history.append(AssistantMessage(content=[TextBlock(text="Test assistant message")]))
        session_data.conversation_history.append(SystemMessage(subtype="init", data={"tool": "test"}))
        session_data.conversation_history.append(ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=3,
            session_id="test-session-456",
            result="Test completed"
        ))
        
        # Convert to dict
        data_dict = session_data.to_dict()
        
        # Verify dict structure
        assert data_dict["session_id"] == "test-session-456"
        assert data_dict["start_time"] == "2025-01-01T12:00:00"
        assert data_dict["last_activity"] == "2025-01-01T12:05:00"
        assert data_dict["working_directory"] == "/test/path"
        assert len(data_dict["conversation_history"]) == 4
        assert data_dict["options"]["model"] == "claude-3-5-sonnet-20241022"
        
        # Convert back from dict
        restored_session = SessionData.from_dict(data_dict)
        
        # Verify restoration
        assert restored_session.session_id == session_data.session_id
        assert restored_session.start_time == session_data.start_time
        assert restored_session.last_activity == session_data.last_activity
        assert restored_session.working_directory == session_data.working_directory
        assert len(restored_session.conversation_history) == 4
        assert restored_session.options.model == "claude-3-5-sonnet-20241022"


class TestSimpleSessionPersistence:
    """Test SimpleSessionPersistence class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def persistence(self, temp_storage):
        """Create SimpleSessionPersistence with temp storage."""
        return SimpleSessionPersistence(temp_storage)

    @pytest.fixture
    def sample_session_data(self):
        """Create sample session data."""
        session_data = SessionData(
            session_id="test-session-789",
            start_time=datetime.now(),
            last_activity=datetime.now(),
            working_directory="/test",
        )
        session_data.add_message(UserMessage(content="Test message"))
        return session_data

    async def test_save_and_load_session(self, persistence, sample_session_data):
        """Test saving and loading sessions."""
        # Save session
        await persistence.save_session(sample_session_data)
        
        # Load session
        loaded_session = await persistence.load_session(sample_session_data.session_id)
        
        assert loaded_session is not None
        assert loaded_session.session_id == sample_session_data.session_id
        assert len(loaded_session.conversation_history) == 1

    async def test_load_nonexistent_session(self, persistence):
        """Test loading nonexistent session returns None."""
        result = await persistence.load_session("nonexistent-session-id")
        assert result is None

    async def test_list_sessions(self, persistence, sample_session_data):
        """Test listing sessions."""
        # Initially empty
        sessions = await persistence.list_sessions()
        assert len(sessions) == 0
        
        # Save a session
        await persistence.save_session(sample_session_data)
        
        # Should now have one session
        sessions = await persistence.list_sessions()
        assert len(sessions) == 1
        assert sessions[0] == sample_session_data.session_id

    async def test_delete_session(self, persistence, sample_session_data):
        """Test deleting sessions."""
        # Save session first
        await persistence.save_session(sample_session_data)
        
        # Verify it exists
        sessions = await persistence.list_sessions()
        assert len(sessions) == 1
        
        # Delete session
        deleted = await persistence.delete_session(sample_session_data.session_id)
        assert deleted is True
        
        # Verify it's gone
        sessions = await persistence.list_sessions()
        assert len(sessions) == 0
        
        # Try to delete again (should return False)
        deleted = await persistence.delete_session(sample_session_data.session_id)
        assert deleted is False

    async def test_corrupted_session_file(self, persistence, temp_storage):
        """Test handling corrupted session files."""
        # Create a corrupted JSON file
        corrupted_file = temp_storage / "corrupted-session.json"
        corrupted_file.write_text("invalid json content")
        
        # Should return None for corrupted file
        result = await persistence.load_session("corrupted-session")
        assert result is None


class TestSessionPersistentClient:
    """Test SessionPersistentClient class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_client(self):
        """Create mock ClaudeSDKClient."""
        with patch('claude_code_sdk.session_persistent_client.ClaudeSDKClient') as mock:
            client_instance = AsyncMock()
            # Set up options as a real ClaudeCodeOptions object to avoid serialization issues
            client_instance.options = ClaudeCodeOptions(model="claude-3-5-sonnet-20241022")
            mock.return_value = client_instance
            yield client_instance

    @pytest.fixture
    def persistent_client(self, mock_client, temp_storage):
        """Create SessionPersistentClient with mocked dependencies."""
        options = ClaudeCodeOptions(model="claude-3-5-sonnet-20241022")
        return SessionPersistentClient(options=options, storage_path=temp_storage)

    def test_initialization(self, persistent_client, mock_client):
        """Test SessionPersistentClient initialization."""
        assert persistent_client._client == mock_client
        assert persistent_client._current_session_id is None
        assert persistent_client._session_data is None

    def test_client_property(self, persistent_client, mock_client):
        """Test client property access."""
        assert persistent_client.client == mock_client

    async def test_connect(self, persistent_client, mock_client):
        """Test connect method."""
        await persistent_client.connect("test prompt")
        mock_client.connect.assert_called_once_with("test prompt")

    async def test_query(self, persistent_client, mock_client):
        """Test query method."""
        await persistent_client.query("test query", "test-session")
        mock_client.query.assert_called_once_with("test query", "test-session")

    async def test_interrupt(self, persistent_client, mock_client):
        """Test interrupt method."""
        await persistent_client.interrupt()
        mock_client.interrupt.assert_called_once()

    async def test_disconnect(self, persistent_client, mock_client):
        """Test disconnect method."""
        await persistent_client.disconnect()
        mock_client.disconnect.assert_called_once()

    async def test_context_manager(self, persistent_client, mock_client):
        """Test async context manager functionality."""
        async with persistent_client as client:
            assert client == persistent_client
            mock_client.connect.assert_called_once_with(None)
        
        mock_client.disconnect.assert_called_once()

    async def test_message_persistence(self, persistent_client):
        """Test automatic message persistence."""
        # Create mock messages
        system_msg = SystemMessage(subtype="init", data={})
        assistant_msg = AssistantMessage(content=[TextBlock(text="Hello!")])
        result_msg = ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="test-session-123",
            result="Done"
        )
        
        # Mock the client's receive_messages to yield these messages
        async def mock_receive_messages():
            yield system_msg
            yield assistant_msg  
            yield result_msg
        
        persistent_client._client.receive_messages = mock_receive_messages
        
        # Collect messages through persistent client
        messages = []
        async for message in persistent_client.receive_messages():
            messages.append(message)
        
        # Verify messages were yielded
        assert len(messages) == 3
        assert messages[0] == system_msg
        assert messages[1] == assistant_msg
        assert messages[2] == result_msg
        
        # Verify session was created when ResultMessage (with session_id) was processed
        assert persistent_client._current_session_id == "test-session-123"
        assert persistent_client._session_data is not None
        # The session is created when the first message with session_id is processed
        # Only that message (and subsequent ones) are saved - previous messages without session context are not
        assert len(persistent_client._session_data.conversation_history) == 1
        assert persistent_client._session_data.conversation_history[0] == result_msg

    async def test_receive_response_stops_at_result(self, persistent_client):
        """Test receive_response stops at ResultMessage."""
        # Create mock messages
        assistant_msg = AssistantMessage(content=[TextBlock(text="Response")])
        result_msg = ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="test-session-456",
            result="Completed"
        )
        extra_msg = SystemMessage(subtype="extra", data={})  # This shouldn't be yielded
        
        # Mock the client's receive_response to properly stop at ResultMessage (like the real implementation does)
        async def mock_receive_response():
            yield assistant_msg
            yield result_msg
            # The real receive_response stops here and doesn't yield extra_msg
        
        persistent_client._client.receive_response = mock_receive_response
        
        # Collect messages
        messages = []
        async for message in persistent_client.receive_response():
            messages.append(message)
        
        # Should stop after ResultMessage (as the underlying client's receive_response does)
        assert len(messages) == 2
        assert messages[0] == assistant_msg
        assert messages[1] == result_msg

    async def test_session_management(self, persistent_client):
        """Test session management methods."""
        # Initially no current session
        assert persistent_client.get_current_session_id() is None
        
        # No sessions saved
        sessions = await persistent_client.list_sessions()
        assert len(sessions) == 0
        
        # Simulate processing a message with session ID
        result_msg = ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="test-session-789",
            result="Test"
        )
        
        await persistent_client._handle_message_persistence(result_msg)
        
        # Should now have current session
        assert persistent_client.get_current_session_id() == "test-session-789"
        
        # Should have one saved session
        sessions = await persistent_client.list_sessions()
        assert len(sessions) == 1
        assert sessions[0] == "test-session-789"
        
        # Should be able to load session data
        session_data = await persistent_client.load_session("test-session-789")
        assert session_data is not None
        assert session_data.session_id == "test-session-789"
        assert len(session_data.conversation_history) == 1
        
        # Should be able to delete session
        deleted = await persistent_client.delete_session("test-session-789")
        assert deleted is True
        
        # Session should be gone
        sessions = await persistent_client.list_sessions()
        assert len(sessions) == 0

    async def test_session_creation_and_loading(self, persistent_client):
        """Test session creation vs loading existing session."""
        session_id = "test-session-999"
        
        # First message creates new session
        first_msg = ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            result="First"
        )
        
        await persistent_client._handle_message_persistence(first_msg)
        original_session = persistent_client._session_data
        
        # Reset client state to simulate new client instance
        persistent_client._current_session_id = None
        persistent_client._session_data = None
        
        # Second message should load existing session
        second_msg = ResultMessage(
            subtype="success",
            duration_ms=1200,
            duration_api_ms=900,
            is_error=False,
            num_turns=2,
            session_id=session_id,
            result="Second"
        )
        
        await persistent_client._handle_message_persistence(second_msg)
        
        # Should have loaded existing session and added new message
        assert persistent_client._session_data is not None
        assert persistent_client._session_data.session_id == session_id
        assert len(persistent_client._session_data.conversation_history) == 2

    async def test_message_without_session_id(self, persistent_client):
        """Test handling messages without session ID."""
        # Message without session_id attribute
        system_msg = SystemMessage(subtype="init", data={})
        
        await persistent_client._handle_message_persistence(system_msg)
        
        # Should not create session data
        assert persistent_client._current_session_id is None
        assert persistent_client._session_data is None

    async def test_final_session_save_on_disconnect(self, persistent_client):
        """Test final session save when disconnecting."""
        # Set up session data
        persistent_client._session_data = SessionData(
            session_id="final-test",
            start_time=datetime.now(),
            last_activity=datetime.now(),
        )
        
        # Mock the client disconnect
        persistent_client._client.disconnect = AsyncMock()
        
        # Disconnect should save session
        await persistent_client.disconnect()
        
        # Verify session was saved and client disconnected
        persistent_client._client.disconnect.assert_called_once()
        
        # Session should exist in storage
        sessions = await persistent_client.list_sessions()
        assert "final-test" in sessions

    async def test_session_id_update_preserves_history(self, persistent_client):
        """Test that session ID updates preserve conversation history."""
        # Start with first message that creates session
        first_msg = ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="session-v1",
            result="First response"
        )
        
        await persistent_client._handle_message_persistence(first_msg)
        
        # Verify initial session setup
        assert persistent_client._current_session_id == "session-v1"
        assert persistent_client._session_data is not None
        assert len(persistent_client._session_data.conversation_history) == 1
        
        # Add more messages to build up history
        assistant_msg = AssistantMessage(content=[TextBlock(text="Building history")])
        await persistent_client._handle_message_persistence(assistant_msg)
        
        user_msg = UserMessage(content="More conversation")
        await persistent_client._handle_message_persistence(user_msg)
        
        # Now we have 3 messages in history
        assert len(persistent_client._session_data.conversation_history) == 3
        original_history = persistent_client._session_data.conversation_history.copy()
        original_start_time = persistent_client._session_data.start_time
        
        # Server sends message with updated session ID (same logical session)
        updated_msg = ResultMessage(
            subtype="success",
            duration_ms=1200,
            duration_api_ms=900,
            is_error=False,
            num_turns=4,
            session_id="session-v2",  # NEW session ID from server
            result="Updated session response"
        )
        
        await persistent_client._handle_message_persistence(updated_msg)
        
        # Verify session ID was updated but history preserved
        assert persistent_client._current_session_id == "session-v2"
        assert persistent_client._session_data.session_id == "session-v2"
        
        # History should be preserved + new message added
        assert len(persistent_client._session_data.conversation_history) == 4
        assert persistent_client._session_data.conversation_history[:3] == original_history
        assert persistent_client._session_data.conversation_history[3] == updated_msg
        
        # Start time should be preserved (same logical session)
        assert persistent_client._session_data.start_time == original_start_time
        
        # Verify new session ID is saved and old one is cleaned up
        sessions = await persistent_client.list_sessions()
        assert "session-v2" in sessions
        assert "session-v1" not in sessions  # Old session should be cleaned up
        
        # Verify the saved session has all the history
        loaded_session = await persistent_client.load_session("session-v2")
        assert loaded_session is not None
        assert len(loaded_session.conversation_history) == 4
        assert loaded_session.start_time == original_start_time

    async def test_multiple_session_id_updates(self, persistent_client):
        """Test multiple session ID updates in sequence."""
        # Start with initial session
        msg1 = ResultMessage(
            subtype="success", duration_ms=1000, duration_api_ms=800,
            is_error=False, num_turns=1, session_id="session-a", result="Response A"
        )
        await persistent_client._handle_message_persistence(msg1)
        
        # First update
        msg2 = ResultMessage(
            subtype="success", duration_ms=1100, duration_api_ms=850,
            is_error=False, num_turns=2, session_id="session-b", result="Response B"
        )
        await persistent_client._handle_message_persistence(msg2)
        
        # Second update  
        msg3 = ResultMessage(
            subtype="success", duration_ms=1200, duration_api_ms=900,
            is_error=False, num_turns=3, session_id="session-c", result="Response C"
        )
        await persistent_client._handle_message_persistence(msg3)
        
        # Verify final state
        assert persistent_client._current_session_id == "session-c"
        assert persistent_client._session_data.session_id == "session-c"
        assert len(persistent_client._session_data.conversation_history) == 3
        
        # Only the final session should exist
        sessions = await persistent_client.list_sessions()
        assert "session-c" in sessions
        assert "session-a" not in sessions
        assert "session-b" not in sessions

    async def test_start_or_resume_session_new(self, persistent_client):
        """Test starting a new session."""
        # Start new session (no session_id provided)
        await persistent_client.start_or_resume_session()
        
        # Should clear any existing session state
        assert persistent_client._current_session_id is None
        assert persistent_client._session_data is None
        
        # Should clear resume option in client options
        if persistent_client._client.options:
            assert persistent_client._client.options.resume is None

    async def test_start_or_resume_session_existing(self, persistent_client):
        """Test resuming an existing session."""
        # First create a session with some data
        session_id = "resume-test-session"
        session_data = SessionData(
            session_id=session_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            working_directory="/test",
        )
        session_data.add_message(UserMessage(content="Previous message"))
        await persistent_client._persistence.save_session(session_data)
        
        # Resume the session
        await persistent_client.start_or_resume_session(session_id)
        
        # Should load the existing session data
        assert persistent_client._current_session_id == session_id
        assert persistent_client._session_data is not None
        assert persistent_client._session_data.session_id == session_id
        assert len(persistent_client._session_data.conversation_history) == 1
        
        # Should set resume option in client options
        assert persistent_client._client.options.resume == session_id

    async def test_start_or_resume_session_nonexistent(self, persistent_client):
        """Test resuming a nonexistent session."""
        nonexistent_id = "nonexistent-session-id"
        
        # Try to resume nonexistent session
        await persistent_client.start_or_resume_session(nonexistent_id)
        
        # Should still set the session_id for resume, even if no local data exists
        # (the server might have the session even if we don't have local data)
        assert persistent_client._current_session_id is None  # No local data loaded
        assert persistent_client._session_data is None
        
        # But should still set resume option for server-side resume
        assert persistent_client._client.options.resume == nonexistent_id

    async def test_start_or_resume_creates_options_if_needed(self, persistent_client):
        """Test that start_or_resume_session creates options if client has none."""
        # Ensure client has no options
        persistent_client._client.options = None
        
        # Resume a session
        await persistent_client.start_or_resume_session("test-session")
        
        # Should create options with resume set
        assert persistent_client._client.options is not None
        assert persistent_client._client.options.resume == "test-session"