"""End-to-end tests for SDK MCP (inline) tools with real Claude API calls.

These tests verify that SDK-created MCP tools work correctly through the full stack,
focusing on tool execution mechanics rather than specific tool functionality.
"""

from typing import Any

import pytest

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)
from claude_code_sdk.types import ToolResultBlock, ToolUseBlock


# Simple test tools
@tool("echo", "Echo back the input text", {"text": str})
async def echo_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Echo back whatever text is provided."""
    return {"content": [{"type": "text", "text": f"Echo: {args['text']}"}]}


@tool("greet", "Greet a person by name", {"name": str})
async def greet_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Greet someone by name."""
    return {"content": [{"type": "text", "text": f"Hello, {args['name']}!"}]}


@pytest.fixture
def sdk_mcp_server():
    """Create a simple SDK MCP server with test tools."""
    return create_sdk_mcp_server(
        name="test-server",
        version="1.0.0",
        tools=[echo_tool, greet_tool],
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_mcp_tool_execution(sdk_mcp_server):
    """Test that SDK MCP tools can be called and executed with allowed_tools."""
    options = ClaudeCodeOptions(
        mcp_servers={"test": sdk_mcp_server},
        allowed_tools=["mcp__test__echo"],
    )
    
    echo_tool_called = False
    echo_result_received = False
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call the mcp__test__echo tool with any text")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name == "mcp__test__echo":
                        echo_tool_called = True
            elif isinstance(message, UserMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        echo_result_received = True
    
    assert echo_tool_called, "mcp__test__echo tool was not called"
    assert echo_result_received, "Tool result was not received"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_mcp_permission_enforcement(sdk_mcp_server):
    """Test that disallowed_tools prevents SDK MCP tool execution."""
    options = ClaudeCodeOptions(
        mcp_servers={"test": sdk_mcp_server},
        disallowed_tools=["mcp__test__echo"],  # Block echo tool
        allowed_tools=["mcp__test__greet"],     # But allow greet
    )
    
    echo_executed = False
    greet_executed = False
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Use the echo tool to echo 'test' and use greet tool to greet 'Alice'")
        
        async for message in client.receive_response():
            if isinstance(message, UserMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        print(block)
                        if "mcp__test__echo" in str(block):
                            echo_executed = True
                        elif "mcp__test__greet" in str(block):
                            greet_executed = True
    
    assert not echo_executed, "Disallowed echo tool was executed"
    assert greet_executed, "Allowed greet tool was not executed"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_mcp_multiple_tools():
    """Test that multiple SDK MCP tools can be called in sequence."""
    server = create_sdk_mcp_server(
        name="multi",
        version="1.0.0",
        tools=[echo_tool, greet_tool],
    )
    
    options = ClaudeCodeOptions(
        mcp_servers={"multi": server},
        allowed_tools=["mcp__multi__echo", "mcp__multi__greet"],
    )
    
    tools_called = []
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("First echo 'test' then greet 'Bob'")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tools_called.append(block.name)
    
    # At least one tool should be called (may be 1 or 2 depending on Claude's approach)
    assert len(tools_called) > 0, "No SDK MCP tools were called"
    # Verify they are SDK MCP tools
    for tool_name in tools_called:
        assert tool_name.startswith("mcp__multi__"), f"Unexpected tool: {tool_name}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_mcp_without_permissions():
    """Test that SDK MCP tools are NOT executed without explicit permissions."""
    server = create_sdk_mcp_server(
        name="noperm",
        version="1.0.0",
        tools=[echo_tool],
    )
    
    # No allowed_tools specified - tools should be blocked
    options = ClaudeCodeOptions(
        mcp_servers={"noperm": server},
    )
    
    tool_result_received = False
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Call the mcp__noperm__echo tool")
        
        async for message in client.receive_response():
            if isinstance(message, UserMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        tool_result_received = True
    
    # SDK MCP tools should NOT execute without explicit allowed_tools
    assert not tool_result_received, "SDK MCP tool was executed without permissions (should be blocked)"