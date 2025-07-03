#!/usr/bin/env python3
"""Example demonstrating MCP (Model Context Protocol) server configuration."""

import anyio

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)


async def with_strict_mcp_config_example():
    """Example using strict MCP configuration."""
    print("=== Strict MCP Config Example ===")

    # This ensures ONLY the MCP servers specified here will be used,
    # ignoring any global or project-level MCP configurations
    options = ClaudeCodeOptions(
        mcp_servers={
            "memory-server": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-memory"],
            }
        },
        strict_mcp_config=True,  # Ignore all file-based MCP configurations
    )

    async for message in query(
        prompt="List the available MCP tools from the memory server",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")
        elif isinstance(message, ResultMessage):
            print(f"\nResult: {message.subtype}")
    print()


async def main():
    """Run the example."""
    await with_strict_mcp_config_example()


if __name__ == "__main__":
    anyio.run(main)