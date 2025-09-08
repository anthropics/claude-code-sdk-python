#!/usr/bin/env python3
"""Debug script to test tool permissions."""

import asyncio
from typing import Any

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)
from claude_code_sdk.types import TextBlock, ToolResultBlock, ToolUseBlock


@tool("add", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Add two numbers together."""
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}]}


async def main():
    """Test permissions without allowed_tools."""
    calculator = create_sdk_mcp_server(
        name="calculator",
        version="2.0.0",
        tools=[add_numbers],
    )
    
    # Create options WITHOUT allowed_tools
    options = ClaudeCodeOptions(
        mcp_servers={"calc": calculator},
        # No allowed_tools specified
    )
    
    print("Testing WITHOUT allowed_tools:")
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Calculate 5 + 3")
        
        async for message in client.receive_response():
            print(f"Message type: {type(message).__name__}")
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        print(f"  TOOL CALLED: {block.name} with {block.input}")
                    elif isinstance(block, TextBlock):
                        print(f"  TEXT: {block.text}")
                    elif isinstance(block, ToolResultBlock):
                        print(f"  TOOL RESULT: {block.content[:200] if block.content else 'None'}")


if __name__ == "__main__":
    asyncio.run(main())