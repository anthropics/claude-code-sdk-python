"""Subprocess transport implementation using Claude Code CLI."""

import json
import os
import shutil
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from subprocess import PIPE
from typing import Any

import anyio
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream

from ..._errors import CLIConnectionError, CLINotFoundError, ProcessError
from ..._errors import CLIJSONDecodeError as SDKJSONDecodeError
from ...types import ClaudeCodeOptions
from . import Transport

logger = logging.getLogger(__name__)


class SubprocessCLITransport(Transport):
    """Subprocess transport using Claude Code CLI."""

    def __init__(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        cli_path: str | Path | None = None,
    ):
        self._prompt = prompt
        self._options = options
        self._cli_path = str(cli_path) if cli_path else self._find_cli()
        self._cwd = str(options.cwd) if options.cwd else None
        self._process: Process | None = None
        self._stdout_stream: TextReceiveStream | None = None
        self._stderr_stream: TextReceiveStream | None = None
        self._timeout = options.timeout
        self._debug = options.debug
        
        if self._debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"Initialized transport with CLI path: {self._cli_path}")

    def _find_cli(self) -> str:
        """Find Claude Code CLI binary."""
        if cli := shutil.which("claude"):
            return cli

        locations = [
            # npm/node locations
            Path.home() / ".npm-global/bin/claude",
            Path("/usr/local/bin/claude"),
            Path.home() / ".local/bin/claude",
            Path.home() / "node_modules/.bin/claude",
            Path.home() / ".yarn/bin/claude",
            # Common global npm locations
            Path("/opt/homebrew/bin/claude"),  # macOS ARM
            Path("/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude"),
            Path.home() / ".nvm/versions/node/*/bin/claude",  # nvm installations
            # Local project installations
            Path("node_modules/.bin/claude"),
            Path("../node_modules/.bin/claude"),
            # Official Claude CLI locations (for future compatibility)
            Path.home() / ".claude/local/claude",
            Path.home() / ".claude/bin/claude",
        ]

        for path in locations:
            # Handle glob patterns (like nvm paths)
            if "*" in str(path):
                import glob
                for expanded_path in glob.glob(str(path)):
                    if Path(expanded_path).exists() and Path(expanded_path).is_file():
                        return expanded_path
            elif path.exists() and path.is_file():
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
            "  SubprocessCLITransport(..., cli_path='/path/to/claude')"
        )

    def _build_command(self) -> list[str]:
        """Build CLI command with arguments."""
        cmd = [self._cli_path, "--output-format", "stream-json", "--verbose"]

        if self._options.system_prompt:
            cmd.extend(["--system-prompt", self._options.system_prompt])

        if self._options.append_system_prompt:
            cmd.extend(["--append-system-prompt", self._options.append_system_prompt])

        if self._options.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self._options.allowed_tools)])

        if self._options.max_turns:
            cmd.extend(["--max-turns", str(self._options.max_turns)])

        if self._options.disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(self._options.disallowed_tools)])

        if self._options.model:
            cmd.extend(["--model", self._options.model])

        if self._options.permission_prompt_tool_name:
            cmd.extend(
                ["--permission-prompt-tool", self._options.permission_prompt_tool_name]
            )

        if self._options.permission_mode:
            cmd.extend(["--permission-mode", self._options.permission_mode])

        if self._options.continue_conversation:
            cmd.append("--continue")

        if self._options.resume:
            cmd.extend(["--resume", self._options.resume])

        if self._options.mcp_servers:
            cmd.extend(
                ["--mcp-config", json.dumps({"mcpServers": self._options.mcp_servers})]
            )

        # Don't add prompt to command - it will be sent via stdin
        return cmd

    async def connect(self) -> None:
        """Start subprocess."""
        if self._process:
            return

        cmd = self._build_command()
        
        if self._debug:
            logger.debug(f"Running command: {' '.join(cmd)}")
            logger.debug(f"Working directory: {self._cwd}")
            
        try:
            self._process = await anyio.open_process(
                cmd,
                stdin=PIPE,  # Need stdin to send prompt
                stdout=PIPE,
                stderr=PIPE,
                cwd=self._cwd,
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "sdk-py"},
            )

            if self._process.stdout:
                self._stdout_stream = TextReceiveStream(self._process.stdout)
            if self._process.stderr:
                self._stderr_stream = TextReceiveStream(self._process.stderr)
            
            # Send the prompt via stdin
            if self._process.stdin:
                if self._debug:
                    logger.debug(f"Sending prompt via stdin: {self._prompt[:100]}...")
                await self._process.stdin.send(self._prompt.encode() + b"\n")
                await self._process.stdin.aclose()  # Close stdin to signal we're done

        except FileNotFoundError as e:
            raise CLINotFoundError(f"Claude Code not found at: {self._cli_path}") from e
        except Exception as e:
            raise CLIConnectionError(f"Failed to start Claude Code: {e}") from e

    async def disconnect(self) -> None:
        """Terminate subprocess."""
        if not self._process:
            return

        if self._process.returncode is None:
            try:
                self._process.terminate()
                timeout = self._timeout or 5.0
                with anyio.fail_after(timeout):
                    await self._process.wait()
            except TimeoutError:
                if self._debug:
                    logger.debug("Process termination timed out, killing...")
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                pass

        self._process = None
        self._stdout_stream = None
        self._stderr_stream = None

    async def send_request(self, messages: list[Any], options: dict[str, Any]) -> None:
        """Not used for CLI transport - args passed via command line."""

    async def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages from CLI."""
        if not self._process or not self._stdout_stream:
            raise CLIConnectionError("Not connected")

        stderr_lines = []
        json_buffer = ""  # Buffer for incomplete JSON

        async def read_stderr() -> None:
            """Read stderr in background."""
            if self._stderr_stream:
                try:
                    async for line in self._stderr_stream:
                        stderr_lines.append(line.strip())
                except anyio.ClosedResourceError:
                    pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(read_stderr)

            try:
                async for line in self._stdout_stream:
                    line_str = line.strip()
                    if not line_str:
                        continue

                    # Handle potential multi-line JSON by buffering
                    if json_buffer:
                        line_str = json_buffer + line_str
                        json_buffer = ""

                    try:
                        data = json.loads(line_str)
                        if self._debug:
                            logger.debug(f"Parsed JSON: {data.get('type', 'unknown')}")
                        try:
                            yield data
                        except GeneratorExit:
                            # Handle generator cleanup gracefully
                            return
                    except json.JSONDecodeError as e:
                        # Check if this might be incomplete JSON
                        if line_str.startswith("{") or line_str.startswith("["):
                            # Check if we have unclosed braces/brackets
                            open_braces = line_str.count("{") - line_str.count("}")
                            open_brackets = line_str.count("[") - line_str.count("]")
                            
                            if open_braces > 0 or open_brackets > 0:
                                # Buffer the incomplete JSON for next iteration
                                if self._debug:
                                    logger.debug(f"Buffering incomplete JSON: {line_str[:100]}...")
                                json_buffer = line_str
                                continue
                            else:
                                # It's complete but invalid JSON
                                if self._debug:
                                    logger.error(f"Invalid JSON: {line_str[:200]}...")
                                raise SDKJSONDecodeError(line_str, e) from e
                        # Skip non-JSON lines
                        if self._debug and line_str:
                            logger.debug(f"Skipping non-JSON line: {line_str[:100]}")
                        continue

            except anyio.ClosedResourceError:
                pass

        # If there's still data in the buffer, try to parse it one more time
        if json_buffer:
            try:
                data = json.loads(json_buffer)
                yield data
            except json.JSONDecodeError as e:
                if json_buffer.startswith("{") or json_buffer.startswith("["):
                    raise SDKJSONDecodeError(json_buffer, e) from e

        await self._process.wait()
        if self._process.returncode is not None and self._process.returncode != 0:
            stderr_output = "\n".join(stderr_lines)
            if stderr_output and "error" in stderr_output.lower():
                raise ProcessError(
                    "CLI process failed",
                    exit_code=self._process.returncode,
                    stderr=stderr_output,
                )

    def is_connected(self) -> bool:
        """Check if subprocess is running."""
        return self._process is not None and self._process.returncode is None
