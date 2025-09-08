"""End-to-end tests for tool permission callbacks with real Claude API calls."""

from typing import Any

import pytest

from claude_code_sdk import (
    ClaudeCodeOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
    create_sdk_mcp_server,
    tool,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_permission_callback_gets_called():
    """Test that can_use_tool callback gets invoked."""
    callback_invocations = []
    
    @tool("test_tool", "A test tool", {"data": str})
    async def test_tool_func(args: dict[str, Any]) -> dict[str, Any]:
        """Test tool."""
        return {"content": [{"type": "text", "text": f"Executed with: {args['data']}"}]}
    
    async def permission_callback(
        tool_name: str,
        input_data: dict,
        context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Track callback invocation."""
        callback_invocations.append(tool_name)
        return PermissionResultAllow()
    
    server = create_sdk_mcp_server(
        name="test",
        version="1.0.0",
        tools=[test_tool_func],
    )
    
    options = ClaudeCodeOptions(
        mcp_servers={"test": server},
        can_use_tool=permission_callback,
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call mcp__test__test_tool with any data")
        
        async for message in client.receive_response():
            pass  # Just consume messages
    
    print('yolo',callback_invocations)
    # Verify callback was invoked
    assert "mcp__test__test_tool" in callback_invocations, "can_use_tool callback should have been invoked"