"""Test multi-turn conversation with StatefulCLITransport."""

import tempfile
from pathlib import Path

import pytest

from claude_code_sdk._internal.transport.subprocess_stateful_cli import (
    StatefulCLITransport,
)
from claude_code_sdk.types import ClaudeCodeOptions


@pytest.mark.asyncio
async def test_stateful_multi_turn_conversation():
    """Test multi-turn conversation functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        # Initialize transport
        transport = StatefulCLITransport(
            session_id=None,
            options=ClaudeCodeOptions(),
            storage_path=storage_path
        )

        # Start session
        session_id = await transport.start_session(resume=False)
        assert session_id is not None
        assert transport.is_session_active()

        # Message 1: Simple math question (avoid problematic patterns)
        response1 = await transport.send_user_message("What is 2+2?", stream=False)

        assert response1.get("subtype") == "success"
        assert "4" in response1.get("result", "")
        assert response1.get("num_turns") == 1

        # Message 2: Test context retention with follow-up math
        response2 = await transport.send_user_message("What about 3+3?", stream=False)

        assert response2.get("subtype") == "success"
        assert "6" in response2.get("result", "")
        assert response2.get("num_turns", 0) > 1  # Should increment from previous turn

        # Message 3: Test further context
        response3 = await transport.send_user_message("What is 4+4?", stream=False)

        assert response3.get("subtype") == "success"
        assert "8" in response3.get("result", "")
        assert response3.get("num_turns", 0) > response2.get("num_turns", 0)  # Should continue incrementing

        # Verify session information
        assert transport.get_session_id() is not None
        assert transport.get_current_session_id() is not None
        assert transport.is_session_active()

        session_data = transport.get_session_data()
        assert session_data is not None
        assert len(session_data.conversation_history) > 0
        assert session_data.message_count > 0

        # End session
        await transport.end_session(save=True)


@pytest.mark.asyncio
async def test_stateful_session_continuity():
    """Test that session IDs are managed correctly across messages."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        transport = StatefulCLITransport(
            session_id=None,
            options=ClaudeCodeOptions(),
            storage_path=storage_path
        )

        # Start session
        await transport.start_session(resume=False)

        # Send first message
        response1 = await transport.send_user_message("Hello", stream=False)
        first_cli_session_id = response1.get("session_id")

        # Send second message
        response2 = await transport.send_user_message("How are you?", stream=False)
        second_cli_session_id = response2.get("session_id")

        # Verify session continuity
        assert first_cli_session_id is not None
        assert second_cli_session_id is not None
        assert response2.get("num_turns", 0) > 1  # Should be continuing conversation

        # Transport should track a CLI session for resume operations
        # Note: CLI may return different session IDs, but transport should track one for resume
        assert transport.get_session_id() is not None

        await transport.end_session(save=True)


@pytest.mark.asyncio
async def test_stateful_session_persistence():
    """Test that session data is properly persisted."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)

        transport = StatefulCLITransport(
            session_id=None,
            options=ClaudeCodeOptions(),
            storage_path=storage_path
        )

        # Start session and send messages
        await transport.start_session(resume=False)

        await transport.send_user_message("My favorite color is blue", stream=False)
        response2 = await transport.send_user_message("What is my favorite color?", stream=False)

        # Verify context is maintained
        assert "blue" in response2.get("result", "").lower()

        # Check session data
        session_data = transport.get_session_data()
        assert session_data is not None
        assert session_data.message_count >= 2
        assert len(session_data.conversation_history) >= 2

        # Save and verify session persists
        await transport.save_session()

        # Verify session file exists
        session_files = list(storage_path.glob("*.json"))
        assert len(session_files) > 0

        await transport.end_session(save=True)
