"""End-to-end tests for agents and setting sources with real Claude API calls."""

import tempfile
from pathlib import Path

import pytest

from claude_code_sdk import (
    AgentDefinition,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    SystemMessage,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_definition():
    """Test that custom agent definitions work."""
    options = ClaudeCodeOptions(
        agents={
            "test-agent": AgentDefinition(
                description="A test agent for verification",
                prompt="You are a test agent. Always respond with 'Test agent activated'",
                tools=["Read"],
                model="sonnet",
            )
        },
        max_turns=1,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 2 + 2?")

        # Check that agent is available in init message
        async for message in client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                agents = message.data.get("agents", {})
                assert (
                    "test-agent" in agents
                ), f"test-agent should be available, got: {agents}"
                agent_data = agents["test-agent"]
                assert agent_data["description"] == "A test agent for verification"
                break


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_setting_sources_user_only():
    """Test that setting_sources=['user'] excludes project settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary project with a slash command
        project_dir = Path(tmpdir)
        commands_dir = project_dir / ".claude" / "commands"
        commands_dir.mkdir(parents=True)

        test_command = commands_dir / "testcmd.md"
        test_command.write_text(
            """---
description: Test command
---

This is a test command.
"""
        )

        # Use setting_sources=["user"] to exclude project settings
        options = ClaudeCodeOptions(
            setting_sources=["user"],
            cwd=project_dir,
            max_turns=1,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query("What is 2 + 2?")

            # Check that project command is NOT available
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    commands = message.data.get("slash_commands", [])
                    assert (
                        "testcmd" not in commands
                    ), f"testcmd should NOT be available with user-only sources, got: {commands}"
                    break


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_setting_sources_project_included():
    """Test that setting_sources=['user', 'project'] includes project settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary project with a slash command
        project_dir = Path(tmpdir)
        commands_dir = project_dir / ".claude" / "commands"
        commands_dir.mkdir(parents=True)

        test_command = commands_dir / "testcmd.md"
        test_command.write_text(
            """---
description: Test command
---

This is a test command.
"""
        )

        # Use setting_sources=["user", "project"] to include project settings
        options = ClaudeCodeOptions(
            setting_sources=["user", "project"],
            cwd=project_dir,
            max_turns=1,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query("What is 2 + 2?")

            # Check that project command IS available
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    commands = message.data.get("slash_commands", [])
                    assert (
                        "testcmd" in commands
                    ), f"testcmd should be available with project sources, got: {commands}"
                    break