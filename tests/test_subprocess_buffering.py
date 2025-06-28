"""Tests for subprocess transport buffering edge cases."""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from claude_code_sdk._errors import CLIJSONDecodeError
from claude_code_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_code_sdk.types import ClaudeCodeOptions


class MockTextReceiveStream:
    """Mock TextReceiveStream for testing."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.index = 0

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if self.index >= len(self.lines):
            raise StopAsyncIteration
        line = self.lines[self.index]
        self.index += 1
        return line


class TestSubprocessBuffering:
    """Test subprocess transport handling of buffered output."""

    def test_multiple_json_objects_on_single_line(self) -> None:
        """Test parsing when multiple JSON objects are concatenated on a single line.

        In some environments, stdout buffering can cause multiple distinct JSON
        objects to be delivered as a single line with embedded newlines.
        """

        async def _test() -> None:
            # Two valid JSON objects separated by a newline character
            json_obj1 = {"type": "message", "id": "msg1", "content": "First message"}
            json_obj2 = {"type": "result", "id": "res1", "status": "completed"}

            # Simulate buffered output where both objects appear on one line
            buffered_line = json.dumps(json_obj1) + "\n" + json.dumps(json_obj2)

            # Create transport
            transport = SubprocessCLITransport(
                prompt="test", options=ClaudeCodeOptions(), cli_path="/usr/bin/claude"
            )

            # Mock the process and streams
            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.wait = AsyncMock(return_value=None)
            transport._process = mock_process

            # Create mock stream that returns the buffered line
            transport._stdout_stream = MockTextReceiveStream([buffered_line])  # type: ignore[assignment]
            transport._stderr_stream = MockTextReceiveStream([])  # type: ignore[assignment]

            # Collect all messages
            messages: list[Any] = []
            async for msg in transport.receive_messages():
                messages.append(msg)

            # Verify both JSON objects were successfully parsed
            assert len(messages) == 2
            assert messages[0]["type"] == "message"
            assert messages[0]["id"] == "msg1"
            assert messages[0]["content"] == "First message"
            assert messages[1]["type"] == "result"
            assert messages[1]["id"] == "res1"
            assert messages[1]["status"] == "completed"

        anyio.run(_test)

    def test_json_with_embedded_newlines(self) -> None:
        """Test parsing JSON objects that contain newline characters in string values."""

        async def _test() -> None:
            # JSON objects with newlines in string values
            json_obj1 = {"type": "message", "content": "Line 1\nLine 2\nLine 3"}
            json_obj2 = {"type": "result", "data": "Some\nMultiline\nContent"}

            buffered_line = json.dumps(json_obj1) + "\n" + json.dumps(json_obj2)

            transport = SubprocessCLITransport(
                prompt="test", options=ClaudeCodeOptions(), cli_path="/usr/bin/claude"
            )

            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.wait = AsyncMock(return_value=None)
            transport._process = mock_process
            transport._stdout_stream = MockTextReceiveStream([buffered_line])
            transport._stderr_stream = MockTextReceiveStream([])

            messages: list[Any] = []
            async for msg in transport.receive_messages():
                messages.append(msg)

            assert len(messages) == 2
            assert messages[0]["content"] == "Line 1\nLine 2\nLine 3"
            assert messages[1]["data"] == "Some\nMultiline\nContent"

        anyio.run(_test)

    def test_multiple_newlines_between_objects(self) -> None:
        """Test parsing with multiple newlines between JSON objects."""

        async def _test() -> None:
            json_obj1 = {"type": "message", "id": "msg1"}
            json_obj2 = {"type": "result", "id": "res1"}

            # Multiple newlines between objects
            buffered_line = json.dumps(json_obj1) + "\n\n\n" + json.dumps(json_obj2)

            transport = SubprocessCLITransport(
                prompt="test", options=ClaudeCodeOptions(), cli_path="/usr/bin/claude"
            )

            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.wait = AsyncMock(return_value=None)
            transport._process = mock_process
            transport._stdout_stream = MockTextReceiveStream([buffered_line])
            transport._stderr_stream = MockTextReceiveStream([])

            messages: list[Any] = []
            async for msg in transport.receive_messages():
                messages.append(msg)

            assert len(messages) == 2
            assert messages[0]["id"] == "msg1"
            assert messages[1]["id"] == "res1"

        anyio.run(_test)

    def test_incomplete_json_across_multiple_lines(self) -> None:
        """Test parsing when JSON is split across multiple lines due to buffering."""

        async def _test() -> None:
            # Large JSON that gets split across multiple lines
            json_obj = {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a very long response that might get split",
                        },
                        {
                            "type": "tool_use",
                            "id": "tool_123",
                            "name": "Read",
                            "input": {"file_path": "/very/long/path/to/file.py"},
                        },
                    ]
                },
            }

            complete_json = json.dumps(json_obj)

            # Split the JSON at an arbitrary point to simulate buffering
            split_point = len(complete_json) // 2
            first_part = complete_json[:split_point]
            second_part = complete_json[split_point:]

            transport = SubprocessCLITransport(
                prompt="test", options=ClaudeCodeOptions(), cli_path="/usr/bin/claude"
            )

            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.wait = AsyncMock(return_value=None)
            transport._process = mock_process

            # Simulate receiving the JSON in two parts
            transport._stdout_stream = MockTextReceiveStream([first_part, second_part])
            transport._stderr_stream = MockTextReceiveStream([])

            messages: list[Any] = []
            async for msg in transport.receive_messages():
                messages.append(msg)

            # Should parse as one complete message
            assert len(messages) == 1
            assert messages[0]["type"] == "assistant"
            assert len(messages[0]["message"]["content"]) == 2

        anyio.run(_test)

    def test_malformed_complete_json_raises_error(self) -> None:
        """Test that malformed but seemingly complete JSON raises an error."""

        async def _test() -> None:
            # JSON that looks complete but is malformed
            malformed_json = '{"type": "message", "invalid": unquoted_value}'

            transport = SubprocessCLITransport(
                prompt="test", options=ClaudeCodeOptions(), cli_path="/usr/bin/claude"
            )

            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.wait = AsyncMock(return_value=None)
            transport._process = mock_process
            transport._stdout_stream = MockTextReceiveStream([malformed_json])
            transport._stderr_stream = MockTextReceiveStream([])

            # Should raise CLIJSONDecodeError for malformed complete JSON
            # The exception will be wrapped in an ExceptionGroup due to anyio task group
            with pytest.raises(Exception) as exc_info:
                messages: list[Any] = []
                async for msg in transport.receive_messages():
                    messages.append(msg)

            # Verify the actual exception is CLIJSONDecodeError
            assert len(exc_info.value.exceptions) == 1
            assert isinstance(exc_info.value.exceptions[0], CLIJSONDecodeError)

        anyio.run(_test)

    def test_no_duplicate_output_when_incomplete_json_followed_by_valid(self) -> None:
        """Test that we don't duplicate output when incomplete JSON is followed by valid JSON."""

        async def _test() -> None:
            # First valid JSON
            valid_json1 = {"type": "user", "message": "first message"}

            # Second valid JSON
            valid_json2 = {"type": "assistant", "message": "second message"}

            # Large JSON that will be incomplete
            large_json = {
                "type": "result",
                "data": "Very large data " * 200,  # Make it large enough to split
                "status": "completed",
            }

            valid_str1 = json.dumps(valid_json1)
            valid_str2 = json.dumps(valid_json2)
            large_str = json.dumps(large_json)

            # Split the large JSON
            split_point = len(large_str) // 2
            large_part1 = large_str[:split_point]
            large_part2 = large_str[split_point:]

            # Create line that has: valid JSON + newline + valid JSON + newline + incomplete large JSON
            combined_line = valid_str1 + "\n" + valid_str2 + "\n" + large_part1

            lines = [
                combined_line,  # First line: 2 valid JSONs + start of large JSON
                large_part2,  # Second line: completion of large JSON
            ]

            transport = SubprocessCLITransport(
                prompt="test", options=ClaudeCodeOptions(), cli_path="/usr/bin/claude"
            )

            mock_process = MagicMock()
            mock_process.returncode = None
            mock_process.wait = AsyncMock(return_value=None)
            transport._process = mock_process
            transport._stdout_stream = MockTextReceiveStream(lines)
            transport._stderr_stream = MockTextReceiveStream([])

            messages: list[Any] = []
            async for msg in transport.receive_messages():
                messages.append(msg)

            # Should have exactly 3 messages: 2 from first line + 1 completed large JSON
            assert len(messages) == 3
            assert messages[0]["type"] == "user"
            assert messages[0]["message"] == "first message"
            assert messages[1]["type"] == "assistant"
            assert messages[1]["message"] == "second message"
            assert messages[2]["type"] == "result"
            assert messages[2]["status"] == "completed"

        anyio.run(_test)
