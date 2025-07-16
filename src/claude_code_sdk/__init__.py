"""Claude SDK for Python."""

import os
from collections.abc import AsyncIterator

from ._errors import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from ._internal.client import InternalClient
from ._internal.transport import Transport
from .types import (
    AssistantMessage,
    ClaudeCodeOptions,
    ContentBlock,
    McpServerConfig,
    Message,
    PermissionMode,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

__version__ = "0.0.14"

__all__ = [
    # Main function
    "query",
    # Transport
    "Transport",
    # Types
    "PermissionMode",
    "McpServerConfig",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    "Message",
    "ClaudeCodeOptions",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Errors
    "ClaudeSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "CLIJSONDecodeError",
]


async def query(
    *, prompt: str, options: ClaudeCodeOptions | None = None, transport: Transport | None = None
) -> AsyncIterator[Message]:
    """
    Query Claude Code.

    Python SDK for interacting with Claude Code.

    Args:
        prompt: The prompt to send to Claude
        options: Optional configuration (defaults to ClaudeCodeOptions() if None).
                 Set options.permission_mode to control tool execution:
                 - 'default': CLI prompts for dangerous tools
                 - 'acceptEdits': Auto-accept file edits
                 - 'bypassPermissions': Allow all tools (use with caution)
                 Set options.cwd for working directory.
        transport: Optional transport implementation. If provided, this will be used
                  instead of the default transport selection based on options.

    Yields:
        Messages from the conversation


    Example:
        ```python
        # Simple usage
        async for message in query(prompt="Hello"):
            print(message)

        # With options
        async for message in query(
            prompt="Hello",
            options=ClaudeCodeOptions(
                system_prompt="You are helpful",
                cwd="/home/user"
            )
        ):
            print(message)

        # With custom transport
        async for message in query(
            prompt="Hello",
            transport=MyCustomTransport()
        ):
            print(message)

        async for message in query(prompt="Hello", transport=transport):
            print(message)
        ```
    """
    if options is None:
        options = ClaudeCodeOptions()

    os.environ["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"

    client = InternalClient()

    async for message in client.process_query(prompt=prompt, options=options, transport=transport):
        yield message
