"""End-to-end tests for hooks with real Claude API calls.

These tests verify that hooks work correctly through the full stack,
focusing on hook execution mechanics rather than specific functionality.
"""

from typing import Any

import pytest

from claude_code_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookJSONOutput,
    HookMatcher,
    Message,
    ResultMessage,
    TextBlock,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pretooluse_hook_blocks_command():
    """Test that PreToolUse hook can block specific bash commands."""
    hook_invocations = []
    
    async def check_bash_command(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Prevent certain bash commands from being executed."""
        tool_name = input_data["tool_name"]
        tool_input = input_data["tool_input"]
        hook_invocations.append(("pre", tool_name))
        
        if tool_name != "Bash":
            return {}
        
        command = tool_input.get("command", "")
        if "foo.sh" in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Command contains invalid pattern: foo.sh",
                }
            }
        
        return {}
    
    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[check_bash_command]),
            ],
        }
    )
    
    async with ClaudeSDKClient(options=options) as client:
        # This command should be blocked
        await client.query("Run the bash command: ./foo.sh --help")
        
        blocked = False
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Check if Claude acknowledges the command was blocked
                        if "blocked" in block.text.lower() or "denied" in block.text.lower():
                            blocked = True
        
        assert "pre" in [inv[0] for inv in hook_invocations], "PreToolUse hook was not invoked"
        assert "Bash" in [inv[1] for inv in hook_invocations], "Hook didn't receive Bash tool name"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pretooluse_hook_allows_safe_command():
    """Test that PreToolUse hook allows safe commands through."""
    hook_invocations = []
    command_executed = False
    
    async def check_bash_command(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Only block dangerous patterns."""
        tool_name = input_data["tool_name"]
        tool_input = input_data["tool_input"]
        hook_invocations.append(tool_name)
        
        if tool_name != "Bash":
            return {}
        
        command = tool_input.get("command", "")
        if "foo.sh" in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Blocked pattern",
                }
            }
        
        return {}  # Allow
    
    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[check_bash_command]),
            ],
        }
    )
    
    async with ClaudeSDKClient(options=options) as client:
        # This safe command should go through
        await client.query("Run the bash command: echo 'Hello from hooks test!'")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if "Hello from hooks test!" in block.text:
                            command_executed = True
        
        assert "Bash" in hook_invocations, "PreToolUse hook was not invoked for Bash"
        assert command_executed, "Safe command should have been executed"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_userpromptsubmit_hook_adds_context():
    """Test that UserPromptSubmit hook can add context to prompts."""
    hook_invocations = []
    
    async def add_custom_context(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Add custom context when user submits prompt."""
        hook_invocations.append("UserPromptSubmit")
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "My favorite color is hot pink",
            }
        }
    
    options = ClaudeAgentOptions(
        hooks={
            "UserPromptSubmit": [
                HookMatcher(matcher=None, hooks=[add_custom_context]),
            ],
        }
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What's my favorite color?")
        
        found_color = False
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if "hot pink" in block.text.lower():
                            found_color = True
        
        assert "UserPromptSubmit" in hook_invocations, "UserPromptSubmit hook was not invoked"
        assert found_color, "Claude should have mentioned 'hot pink' based on hook context"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multiple_hooks_same_event():
    """Test that multiple hooks can be registered for the same event."""
    hook_invocations = []
    
    async def hook1(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        hook_invocations.append("hook1")
        return {}
    
    async def hook2(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        hook_invocations.append("hook2")
        return {}
    
    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[hook1, hook2]),
            ],
        }
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Run: echo 'test'")
        
        async for message in client.receive_response():
            pass  # Just consume messages
        
        # Both hooks should have been invoked
        assert "hook1" in hook_invocations, "First hook was not invoked"
        assert "hook2" in hook_invocations, "Second hook was not invoked"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_hook_with_matcher_pattern():
    """Test that hook matchers filter by tool name pattern."""
    write_hook_called = False
    bash_hook_called = False
    
    async def write_hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        nonlocal write_hook_called
        write_hook_called = True
        return {}
    
    async def bash_hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        nonlocal bash_hook_called
        bash_hook_called = True
        return {}
    
    options = ClaudeAgentOptions(
        allowed_tools=["Write", "Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Write", hooks=[write_hook]),
                HookMatcher(matcher="Bash", hooks=[bash_hook]),
            ],
        }
    )
    
    async with ClaudeSDKClient(options=options) as client:
        # Request both Write and Bash operations
        await client.query("Write 'test' to /tmp/hook_test.txt and then run: echo 'done'")
        
        async for message in client.receive_response():
            pass  # Just consume messages
        
        # Check that the right hooks were called for the right tools
        assert write_hook_called, "Write hook should have been called for Write tool"
        assert bash_hook_called, "Bash hook should have been called for Bash tool"