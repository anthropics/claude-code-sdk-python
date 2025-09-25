#!/usr/bin/env python3
"""Example demonstrating setting sources control.

This example shows how to use the setting_sources option to control which
settings are loaded, including custom slash commands, agents, and other
configurations.

Setting sources determine where Claude Code loads configurations from:
- "user": Global user settings (~/.claude/)
- "project": Project-level settings (.claude/ in project)
- "local": Local gitignored settings (.claude-local/)

By controlling which sources are loaded, you can:
- Exclude project-specific slash commands
- Use only your personal configurations
- Create isolated environments with minimal settings

Usage:
./examples/setting_sources.py - List the examples
./examples/setting_sources.py all - Run all examples
./examples/setting_sources.py user_only - Run a specific example
"""

import asyncio
import sys
from pathlib import Path

from claude_code_sdk import (
    ClaudeCodeOptions,
    ClaudeSDKClient,
    SystemMessage,
)


def extract_slash_commands(msg: SystemMessage) -> list[str]:
    """Extract slash command names from system message."""
    if msg.subtype == "init":
        commands = msg.data.get("slash_commands", [])
        return commands
    return []


async def example_user_only():
    """Load only user-level settings, excluding project settings."""
    print("=== User Settings Only Example ===")
    print("Setting sources: ['user']")
    print("Expected: Project slash commands (like /commit) will NOT be available\n")

    # Use the SDK repo directory which has .claude/commands/commit.md
    sdk_dir = Path(__file__).parent.parent

    options = ClaudeCodeOptions(
        setting_sources=["user"],
        cwd=sdk_dir,
    )

    async with ClaudeSDKClient(options=options) as client:
        # Send a simple query
        await client.query("What is 2 + 2?")

        # Check the initialize message for available commands
        async for msg in client.receive_response():
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                commands = extract_slash_commands(msg)
                print(f"Available slash commands: {commands}")
                if "commit" in commands:
                    print("❌ /commit is available (unexpected)")
                else:
                    print("✓ /commit is NOT available (expected)")
                break

    print()


async def example_project_and_user():
    """Load both project and user settings."""
    print("=== Project + User Settings Example ===")
    print("Setting sources: ['user', 'project']")
    print("Expected: Project slash commands (like /commit) WILL be available\n")

    sdk_dir = Path(__file__).parent.parent

    options = ClaudeCodeOptions(
        setting_sources=["user", "project"],
        cwd=sdk_dir,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 2 + 2?")

        async for msg in client.receive_response():
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                commands = extract_slash_commands(msg)
                print(f"Available slash commands: {commands}")
                if "commit" in commands:
                    print("✓ /commit is available (expected)")
                else:
                    print("❌ /commit is NOT available (unexpected)")
                break

    print()


async def example_empty_sources():
    """Load no settings at all."""
    print("=== No Settings Example ===")
    print("Setting sources: []\n")

    sdk_dir = Path(__file__).parent.parent

    options = ClaudeCodeOptions(
        setting_sources=[],
        cwd=sdk_dir,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 2 + 2?")

        async for msg in client.receive_response():
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                commands = extract_slash_commands(msg)
                print(f"Available slash commands: {commands}")
                break

    print()


async def main():
    """Run all examples or a specific example based on command line argument."""
    examples = {
        "user_only": example_user_only,
        "project_and_user": example_project_and_user,
        "empty_sources": example_empty_sources,
    }

    if len(sys.argv) < 2:
        print("Usage: python setting_sources.py <example_name>")
        print("\nAvailable examples:")
        print("  all - Run all examples")
        for name in examples:
            print(f"  {name}")
        sys.exit(0)

    example_name = sys.argv[1]

    if example_name == "all":
        for example in examples.values():
            await example()
            print("-" * 50 + "\n")
    elif example_name in examples:
        await examples[example_name]()
    else:
        print(f"Error: Unknown example '{example_name}'")
        print("\nAvailable examples:")
        print("  all - Run all examples")
        for name in examples:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Claude SDK Setting Sources Examples...")
    print("=" * 50 + "\n")
    asyncio.run(main())