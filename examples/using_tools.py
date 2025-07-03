#!/usr/bin/env python3
"""Example demonstrating how to use tools with Claude Code SDK."""

import anyio

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)


async def with_tools_example():
    """Example using tools."""
    print("=== With Tools Example ===")

    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write"],
        system_prompt="You are a helpful file assistant.",
    )

    async for message in query(
        prompt="Create a file called hello.txt with 'Hello, World!' in it",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")
        elif isinstance(message, ResultMessage) and message.total_cost_usd > 0:
            print(f"\nCost: ${message.total_cost_usd:.4f}")
    print()


async def main():
    """Run the example."""
    await with_tools_example()


if __name__ == "__main__":
    anyio.run(main)