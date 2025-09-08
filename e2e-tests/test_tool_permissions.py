"""End-to-end tests for tool permission callbacks with real Claude API calls.

These tests verify that tool permission callbacks work correctly by tracking
actual tool execution rather than just protocol messages.
"""

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
async def test_permission_callback_allows_tools():
    """Test that callback returning Allow permits tool execution."""
    executions = []
    
    @tool("test_tool", "A test tool", {"data": str})
    async def test_tool_func(args: dict[str, Any]) -> dict[str, Any]:
        """Test tool that tracks execution."""
        executions.append("test_tool")
        return {"content": [{"type": "text", "text": f"Executed with: {args['data']}"}]}
    
    async def allow_callback(
        tool_name: str,
        input_data: dict,
        context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Always allow tools."""
        return PermissionResultAllow()
    
    server = create_sdk_mcp_server(
        name="test",
        version="1.0.0",
        tools=[test_tool_func],
    )
    
    options = ClaudeCodeOptions(
        mcp_servers={"test": server},
        can_use_tool=allow_callback,
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call mcp__test__test_tool with any data")
        
        async for message in client.receive_response():
            pass  # Just consume messages
    
    # Verify tool was executed
    assert "test_tool" in executions, "Tool should have been executed with allow callback"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_permission_callback_denies_tools():
    """Test that callback returning Deny prevents tool execution."""
    executions = []
    
    @tool("test_tool", "A test tool", {"data": str})
    async def test_tool_func(args: dict[str, Any]) -> dict[str, Any]:
        """Test tool that tracks execution."""
        executions.append("test_tool")
        return {"content": [{"type": "text", "text": f"Executed with: {args['data']}"}]}
    
    async def deny_callback(
        tool_name: str,
        input_data: dict,
        context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Always deny tools."""
        return PermissionResultDeny(message="Permission denied by callback")
    
    server = create_sdk_mcp_server(
        name="test",
        version="1.0.0",
        tools=[test_tool_func],
    )
    
    options = ClaudeCodeOptions(
        mcp_servers={"test": server},
        can_use_tool=deny_callback,
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call mcp__test__test_tool with any data")
        
        async for message in client.receive_response():
            pass  # Just consume messages
    
    # Verify tool was NOT executed
    assert "test_tool" not in executions, "Tool should NOT have been executed with deny callback"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_permission_callback_modifies_input():
    """Test that callback can modify tool inputs before execution."""
    executions = []
    received_inputs = []
    
    @tool("test_tool", "A test tool", {"data": str})
    async def test_tool_func(args: dict[str, Any]) -> dict[str, Any]:
        """Test tool that tracks execution and inputs."""
        executions.append("test_tool")
        received_inputs.append(args.get("data"))
        return {"content": [{"type": "text", "text": f"Executed with: {args['data']}"}]}
    
    async def modify_callback(
        tool_name: str,
        input_data: dict,
        context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Modify input before allowing."""
        # Change the data to something else
        modified_input = input_data.copy()
        modified_input["data"] = "MODIFIED_BY_CALLBACK"
        return PermissionResultAllow(updated_input=modified_input)
    
    server = create_sdk_mcp_server(
        name="test",
        version="1.0.0",
        tools=[test_tool_func],
    )
    
    options = ClaudeCodeOptions(
        mcp_servers={"test": server},
        can_use_tool=modify_callback,
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call mcp__test__test_tool with data='original'")
        
        async for message in client.receive_response():
            pass  # Just consume messages
    
    # Verify tool was executed with modified input
    assert "test_tool" in executions, "Tool should have been executed"
    assert "MODIFIED_BY_CALLBACK" in received_inputs, "Tool should receive modified input"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_selective_permission_by_tool_name():
    """Test that callback can selectively allow/deny based on tool name."""
    executions = []
    
    @tool("safe_tool", "A safe tool", {"data": str})
    async def safe_tool_func(args: dict[str, Any]) -> dict[str, Any]:
        """Safe tool that tracks execution."""
        executions.append("safe_tool")
        return {"content": [{"type": "text", "text": "Safe tool executed"}]}
    
    @tool("dangerous_tool", "A dangerous tool", {"data": str})
    async def dangerous_tool_func(args: dict[str, Any]) -> dict[str, Any]:
        """Dangerous tool that tracks execution."""
        executions.append("dangerous_tool")
        return {"content": [{"type": "text", "text": "Dangerous tool executed"}]}
    
    async def selective_callback(
        tool_name: str,
        input_data: dict,
        context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Allow safe tools, deny dangerous ones."""
        if "safe" in tool_name:
            return PermissionResultAllow()
        else:
            return PermissionResultDeny(message="Dangerous tool blocked")
    
    server = create_sdk_mcp_server(
        name="test",
        version="1.0.0",
        tools=[safe_tool_func, dangerous_tool_func],
    )
    
    options = ClaudeCodeOptions(
        mcp_servers={"test": server},
        can_use_tool=selective_callback,
    )
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call both mcp__test__safe_tool and mcp__test__dangerous_tool")
        
        async for message in client.receive_response():
            pass  # Just consume messages
    
    # Verify selective execution
    assert "safe_tool" in executions, "Safe tool should have been executed"
    assert "dangerous_tool" not in executions, "Dangerous tool should have been blocked"