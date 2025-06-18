"""Subprocess transport implementation using Claude Code CLI."""

import json
import os
import shutil
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


class JSONStreamParser:
    """Handles parsing of potentially incomplete JSON streams."""

    def __init__(self):
        self._buffer = ""

    def add_data(self, data: str) -> list[dict[str, Any]]:
        """Add new data to buffer and return any complete JSON objects."""
        self._buffer += data
        return self._extract_complete_objects()

    def _extract_complete_objects(self) -> list[dict[str, Any]]:
        """Extract all complete JSON objects from the buffer."""
        objects = []

        # Handle newline-separated JSON objects
        lines = self._buffer.split('\n')

        # Keep the last line in buffer
        self._buffer = lines[-1] if lines else ""

        # Process all complete lines
        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue

            if parsed_obj := self._try_parse_single_line(line):
                objects.append(parsed_obj)

        # Try to parse remaining buffer for complete objects
        while self._buffer:
            parsed_obj, remaining = self._try_parse_partial_buffer()
            if parsed_obj:
                objects.append(parsed_obj)
                self._buffer = remaining
            else:
                break

        return objects

    def _try_parse_single_line(self, line: str) -> dict[str, Any] | None:
        """Try to parse a single line as JSON."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # If single line fails, add to buffer for partial parsing
            self._buffer = line + "\n" + self._buffer
            return None

    def _try_parse_partial_buffer(self) -> tuple[dict[str, Any] | None, str]:
        """Try to extract complete JSON object from partial buffer."""
        buffer = self._buffer.strip()
        if not buffer:
            return None, ""

        # Quick attempt at full parse
        try:
            return json.loads(buffer), ""
        except json.JSONDecodeError:
            pass

        # Try to find complete JSON object by tracking braces
        complete_object = self._find_complete_json_object(buffer)
        if complete_object:
            try:
                parsed = json.loads(complete_object)
                remaining = buffer[len(complete_object):].strip()
                return parsed, remaining
            except json.JSONDecodeError:
                pass

        return None, buffer

    def _find_complete_json_object(self, text: str) -> str | None:
        """Find the first complete JSON object in text using brace counting."""
        if not text.startswith('{'):
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[:i + 1]

        return None


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

        cmd.extend(["--print", self._prompt])
        return cmd

    async def connect(self) -> None:
        """Start subprocess."""
        if self._process:
            return

        cmd = self._build_command()
        try:
            self._process = await anyio.open_process(
                cmd,
                stdin=None,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self._cwd,
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "sdk-py"},
            )

            if self._process.stdout:
                self._stdout_stream = TextReceiveStream(self._process.stdout)
            if self._process.stderr:
                self._stderr_stream = TextReceiveStream(self._process.stderr)

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
                with anyio.fail_after(5.0):
                    await self._process.wait()
            except TimeoutError:
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
        json_parser = JSONStreamParser()

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

                    # Parse potentially incomplete JSON stream
                    try:
                        complete_objects = json_parser.add_data(line_str)
                        for json_obj in complete_objects:
                            yield json_obj
                    except Exception as e:
                        # If parsing fails completely, try simple fallback
                        if line_str.startswith("{") or line_str.startswith("["):
                            try:
                                data = json.loads(line_str)
                                yield data
                            except json.JSONDecodeError as json_err:
                                raise SDKJSONDecodeError(line_str, json_err) from json_err

            except anyio.ClosedResourceError:
                pass

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
