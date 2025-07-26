"""Stateful subprocess transport implementation for persistent Claude Code CLI sessions."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from subprocess import PIPE
from typing import Any

import anyio

from ..._errors import CLIConnectionError, CLINotFoundError, ProcessError
from ..._errors import CLIJSONDecodeError as SDKJSONDecodeError
from ...types import ClaudeCodeOptions
from ..session_persistence import ConversationMessage, SessionData, SessionPersistence

logger = logging.getLogger(__name__)


class StatefulCLITransport:
    """
    Stateful transport that uses individual CLI calls with session resumption.

    This transport makes separate claude CLI calls for each message, using
    --resume to maintain conversation continuity. The CLI handles all
    conversation state internally.
    """

    def __init__(
        self,
        session_id: str | None = None,
        options: ClaudeCodeOptions | None = None,
        cli_path: str | Path | None = None,
        storage_path: Path | str | None = None,
    ):
        self._original_session_id = session_id  # Track original session for resume
        self._current_session_id = session_id  # May change with each response
        self._options = options or ClaudeCodeOptions()
        self._cli_path = str(cli_path) if cli_path else self._find_cli()
        self._cwd = str(self._options.cwd) if self._options.cwd else None
        self._persistence = SessionPersistence(storage_path)
        self._session_data: SessionData | None = None
        self._session_active = False
        self._has_sent_messages = False  # Track if we've sent any messages

    def _find_cli(self) -> str:
        """Find Claude Code CLI binary."""
        if cli := shutil.which("claude"):
            return cli

        locations = [
            Path.home() / ".npm-global/bin/claude",
            Path("/usr/local/bin/claude"),
            Path.home() / ".local/bin/claude",
            Path.home() / "node_modules/.bin/claude",
            Path.home() / ".yarn/bin/claude",
        ]

        for path in locations:
            if path.exists() and path.is_file():
                return str(path)

        node_installed = shutil.which("node") is not None

        if not node_installed:
            error_msg = "Claude Code requires Node.js, which is not installed.\n\n"
            error_msg += "Install Node.js from: https://nodejs.org/\n"
            error_msg += "\nAfter installing Node.js, install Claude Code:\n"
            error_msg += "  npm install -g @anthropic-ai/claude-code"
            raise CLINotFoundError(error_msg)

        raise CLINotFoundError(
            "Claude Code not found. Install with:\n"
            "  npm install -g @anthropic-ai/claude-code\n"
            "\nIf already installed locally, try:\n"
            '  export PATH="$HOME/node_modules/.bin:$PATH"\n'
            "\nOr specify the path when creating transport:\n"
            "  StatefulCLITransport(..., cli_path='/path/to/claude')"
        )

    async def start_session(self, resume: bool = True) -> str:
        """
        Start or resume a session.

        Args:
            resume: Whether to try resuming existing session

        Returns:
            str: Session ID for the conversation
        """
        if self._session_active:
            return self._current_session_id or ""

        # Try to load existing session metadata if resuming
        if resume and self._original_session_id:
            self._session_data = await self._persistence.load_session(
                self._original_session_id
            )
            if self._session_data:
                logger.info(f"Loaded existing session: {self._original_session_id}")

        # Create new session metadata if needed
        if not self._session_data:
            now = datetime.now()
            session_id = self._original_session_id or str(uuid.uuid4())
            self._session_data = SessionData(
                session_id=session_id,
                start_time=now,
                last_activity=now,
                options=self._options,
                working_directory=self._cwd,
            )
            await self._persistence.save_session(self._session_data)

            # Set both IDs to the new session for first-time creation
            if not self._original_session_id:
                self._original_session_id = session_id
                self._current_session_id = session_id

        self._session_active = True
        return self._current_session_id or self._original_session_id or ""

    async def end_session(self, save: bool = True) -> None:
        """End the current session."""
        if save and self._session_data:
            await self._persistence.save_session(self._session_data)

        self._session_active = False
        logger.info(f"Ended session: {self._original_session_id}")

    async def send_user_message(
        self, content: str, stream: bool = False
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Send a user message and get response.

        Args:
            content: User message content
            stream: If True, return streaming response; if False, return simple response

        Returns:
            dict: Single response object if stream=False
            AsyncIterator[dict]: Streaming responses if stream=True
        """
        if not self._session_active:
            raise CLIConnectionError("Session not active")

        if stream:
            return self._send_user_message_streaming(content)
        else:
            return await self.get_simple_response(content)

    async def _send_user_message_streaming(
        self, content: str
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Send a user message and get streaming response.

        Args:
            content: User message content

        Yields:
            dict: Parsed JSON messages from CLI stream-json output
        """
        # Build CLI command for streaming
        cmd = self._build_message_command(content, stream=True)

        try:
            # Execute CLI command and parse streaming response
            async for message in self._execute_streaming_command(cmd):
                # Update session tracking from response
                if message.get("type") == "result":
                    # Mark that we've sent messages
                    self._has_sent_messages = True

                    # Update current session ID (CLI may change it)
                    if "session_id" in message:
                        # Always update the original session ID to what CLI returns
                        # This ensures we use the CLI's session for subsequent resume operations
                        self._original_session_id = message["session_id"]
                        self._current_session_id = message["session_id"]

                    # Update session metadata
                    if self._session_data:
                        self._session_data.last_activity = datetime.now()

                        # Add user message to history
                        user_msg = ConversationMessage(
                            role="user",
                            content=content,
                            timestamp=datetime.now(),
                            message_type="text",
                        )
                        self._session_data.add_message(user_msg)

                        # Add assistant response to history
                        if "result" in message:
                            assistant_msg = ConversationMessage(
                                role="assistant",
                                content=message["result"],
                                timestamp=datetime.now(),
                                message_type="text",
                                usage=message.get("usage"),
                            )
                            self._session_data.add_message(assistant_msg)

                yield message

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise CLIConnectionError(f"Failed to send message: {e}") from e

    def _build_message_command(self, content: str, stream: bool = False) -> list[str]:
        """Build CLI command for sending a message."""
        cmd = [self._cli_path, "-p", content]

        # Add session resumption if we have a session and have sent messages
        if self._original_session_id and self._has_sent_messages:
            cmd.extend(["--resume", self._original_session_id])

        # Choose output format based on stream parameter
        if stream:
            # Use stream-json for real-time tool visibility
            cmd.extend(["--output-format", "stream-json", "--verbose"])
        else:
            # Use simple JSON for direct response
            cmd.extend(["--output-format", "json"])

        # Add other options
        if self._options.system_prompt:
            cmd.extend(["--system-prompt", self._options.system_prompt])

        if self._options.append_system_prompt:
            cmd.extend(["--append-system-prompt", self._options.append_system_prompt])

        if self._options.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._options.allowed_tools)])

        if self._options.disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(self._options.disallowed_tools)])

        if self._options.model:
            cmd.extend(["--model", self._options.model])

        if self._options.permission_mode:
            cmd.extend(["--permission-mode", self._options.permission_mode])

        if self._options.max_turns:
            cmd.extend(["--max-turns", str(self._options.max_turns)])

        if self._options.mcp_servers:
            cmd.extend(
                ["--mcp-config", json.dumps({"mcpServers": self._options.mcp_servers})]
            )

        return cmd

    async def _execute_streaming_command(
        self, cmd: list[str]
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute CLI command and parse streaming JSON output."""
        try:
            process = await anyio.open_process(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self._cwd,
                env=os.environ,
            )

            if not process.stdout:
                raise CLIConnectionError("No stdout from CLI process")

            # Parse streaming JSON output
            json_buffer = ""
            async for line in process.stdout:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                # Handle multiple JSON objects on one line
                json_lines = line_str.split("\n")
                for json_line in json_lines:
                    json_line = json_line.strip()
                    if not json_line:
                        continue

                    # Accumulate partial JSON
                    json_buffer += json_line

                    try:
                        data = json.loads(json_buffer)
                        json_buffer = ""  # Reset buffer on successful parse
                        yield data
                    except json.JSONDecodeError:
                        # Continue accumulating - might be partial JSON
                        continue

            # Wait for process completion
            returncode = await process.wait()

            # Handle errors
            if returncode != 0:
                stderr_output = ""
                if process.stderr:
                    stderr_data = await process.stderr.receive()
                    stderr_output = stderr_data.decode("utf-8")

                raise ProcessError(
                    f"CLI command failed with exit code {returncode}",
                    exit_code=returncode,
                    stderr=stderr_output,
                )

        except FileNotFoundError as e:
            if self._cwd and not Path(self._cwd).exists():
                raise CLIConnectionError(
                    f"Working directory does not exist: {self._cwd}"
                ) from e
            raise CLINotFoundError(f"Claude Code not found at: {self._cli_path}") from e
        except Exception as e:
            raise CLIConnectionError(f"Failed to execute CLI command: {e}") from e

    def get_session_id(self) -> str | None:
        """Get the original session ID."""
        return self._original_session_id

    def get_current_session_id(self) -> str | None:
        """Get the current session ID (may change with each response)."""
        return self._current_session_id

    def is_session_active(self) -> bool:
        """Check if session is active."""
        return self._session_active

    def get_session_data(self) -> SessionData | None:
        """Get current session data."""
        return self._session_data

    async def save_session(self) -> None:
        """Manually save session data."""
        if self._session_data:
            await self._persistence.save_session(self._session_data)

    async def get_simple_response(self, content: str) -> dict[str, Any]:
        """
        Get a simple JSON response without streaming.

        Args:
            content: User message content

        Returns:
            dict: Complete response as single JSON object
        """
        if not self._session_active:
            raise CLIConnectionError("Session not active")

        # Build command with JSON output format using the unified method
        cmd = self._build_message_command(content, stream=False)

        # Add delay before resume operations to allow CLI session to be saved
        if self._has_sent_messages:
            await asyncio.sleep(1.5)  # 1.5s delay for resume operations

        # Retry logic for operations that may fail due to server-side issues
        max_retries = 4  # Aggressive retry for all operations due to server-side intermittency
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                # Add progressive delay for retries
                if attempt > 0:
                    retry_delay = 1.0 + (attempt * 0.5)  # Progressive delay: 1.5s, 2.0s, 2.5s
                    await asyncio.sleep(retry_delay)
                    logger.warning(
                        f"Retrying CLI command (attempt {attempt + 1}/{max_retries + 1}) after {retry_delay}s delay"
                    )

                # Use subprocess for simple response (more reliable than anyio for this case)
                # Increase timeout for retry attempts to handle server-side delays
                timeout = 120 + (attempt * 30)  # Progressive timeout: 120s, 150s, 180s, 210s
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self._cwd,
                    env=os.environ,
                    timeout=timeout,
                )

                if result.returncode != 0:
                    raise ProcessError(
                        f"CLI command failed with exit code {result.returncode}",
                        exit_code=result.returncode,
                        stderr=result.stderr,
                    )

                # Parse JSON response
                response_text = result.stdout.strip()
                if not response_text:
                    raise SDKJSONDecodeError(
                        "Empty response from CLI", ValueError("No content")
                    )

                response_data: dict[str, Any] = json.loads(response_text)

                # Debug logging for resume issues
                logger.debug(f"CLI Response: {response_data}")
                if self._has_sent_messages:
                    logger.debug(
                        f"Resume command used session ID: {self._original_session_id}"
                    )

                # Check for CLI execution errors - retry for server-side issues
                if response_data.get("subtype") == "error_during_execution":
                    error_msg = response_data.get("result", "CLI execution error")
                    if attempt < max_retries:
                        logger.warning(
                            f"Server-side error on attempt {attempt + 1}/{max_retries + 1}: {error_msg}"
                        )
                        logger.warning(f"Command: {' '.join(cmd)}")
                        logger.warning(f"Full response: {response_data}")
                        last_error = CLIConnectionError(
                            f"CLI execution failed: {error_msg}"
                        )
                        continue  # Retry
                    else:
                        logger.error(f"CLI execution failed after {max_retries + 1} attempts")
                        logger.error(f"Command: {' '.join(cmd)}")
                        logger.error(f"Final response: {response_data}")
                        raise CLIConnectionError(f"CLI execution failed after {max_retries + 1} attempts: {error_msg}")

                # Success - break out of retry loop
                break

            except CLIConnectionError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"CLI connection error on attempt {attempt + 1}/{max_retries + 1}: {e}"
                    )
                    continue
                else:
                    raise
            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"CLI timeout on attempt {attempt + 1}/{max_retries + 1} after {e.timeout}s, retrying with longer timeout"
                    )
                    continue
                else:
                    raise CLIConnectionError(f"CLI command timed out after {e.timeout} seconds on final attempt") from e
            except Exception as e:
                # For other exceptions, don't retry
                last_error = e
                raise

        # Mark that we've sent messages and update session tracking
        self._has_sent_messages = True

        # Update session tracking and metadata
        if "session_id" in response_data:
            # Always update the original session ID to what CLI returns
            # This ensures we use the CLI's session for subsequent resume operations
            self._original_session_id = response_data["session_id"]
            self._current_session_id = response_data["session_id"]

        # Update session metadata
        if self._session_data:
            self._session_data.last_activity = datetime.now()

            # Add user message to history
            user_msg = ConversationMessage(
                role="user",
                content=content,
                timestamp=datetime.now(),
                message_type="text",
            )
            self._session_data.add_message(user_msg)

            # Add assistant response to history
            if "result" in response_data:
                assistant_msg = ConversationMessage(
                    role="assistant",
                    content=response_data["result"],
                    timestamp=datetime.now(),
                    message_type="text",
                    usage=response_data.get("usage"),
                )
                self._session_data.add_message(assistant_msg)

        return response_data
