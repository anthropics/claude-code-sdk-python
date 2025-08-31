"""Integration tests for SDK MCP server support.

This test file verifies that SDK MCP servers work correctly through the full stack,
matching the TypeScript SDK test/sdk.test.ts pattern.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from claude_code_sdk import (
    ClaudeCodeOptions,
    create_sdk_mcp_server,
    query,
    tool,
)


@pytest.mark.asyncio
async def test_sdk_mcp_server_integration():
    """Full integration test matching TypeScript SDK test/sdk.test.ts."""
    # Track tool executions
    tool_executions: List[Dict[str, Any]] = []
    
    # Create SDK MCP server with multiple tools
    @tool("greet_user", "Greets a user by name", {"name": str})
    async def greet_user(args: Dict[str, Any]) -> Dict[str, Any]:
        tool_executions.append({"name": "greet_user", "args": args})
        return {
            "content": [
                {"type": "text", "text": f"Hello, {args['name']}!"}
            ]
        }
    
    @tool("add_numbers", "Adds two numbers", {"a": float, "b": float})
    async def add_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
        tool_executions.append({"name": "add_numbers", "args": args})
        result = args["a"] + args["b"]
        return {
            "content": [
                {"type": "text", "text": f"The sum of {args['a']} and {args['b']} is {result}"}
            ]
        }
    
    server = create_sdk_mcp_server(
        name="test-sdk-server",
        version="1.0.0",
        tools=[greet_user, add_numbers]
    )
    
    # Execute through query() API
    messages = []
    async for msg in query(
        prompt="Please greet Alice and then add 5 + 3",
        options=ClaudeCodeOptions(
            mcp_servers={"test": server},
            max_turns=1
        )
    ):
        messages.append(msg)
    
    # Verify we got messages
    assert len(messages) > 0
    
    # Check for init message
    init_msg = next((m for m in messages if hasattr(m, 'subtype') and m.subtype == 'init'), None)
    assert init_msg is not None, "Should have init message"
    
    # Check for result message
    result_msg = next((m for m in messages if hasattr(m, 'subtype') and m.subtype == 'result'), None)
    assert result_msg is not None, "Should have result message"
    
    # Verify tools were executed (this would happen if Claude actually calls them)
    # Note: In a real test, we'd need Claude to actually call the tools
    # For now, we're just verifying the setup works


@pytest.mark.asyncio
async def test_tool_creation():
    """Test that tools can be created with proper schemas."""
    @tool("echo", "Echo input", {"input": str})
    async def echo_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": args["input"]}
    
    # Verify tool was created
    assert echo_tool.name == "echo"
    assert echo_tool.description == "Echo input"
    assert echo_tool.input_schema == {"input": str}
    assert callable(echo_tool.handler)
    
    # Test the handler works
    result = await echo_tool.handler({"input": "test"})
    assert result == {"output": "test"}


@pytest.mark.asyncio
async def test_error_handling():
    """Test that tool errors are properly handled."""
    @tool("fail", "Always fails", {})
    async def fail_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Expected error")
    
    # Verify the tool raises an error when called
    with pytest.raises(ValueError, match="Expected error"):
        await fail_tool.handler({})


@pytest.mark.asyncio
async def test_mixed_servers():
    """Test that SDK and external MCP servers can work together."""
    # Create an SDK server
    @tool("sdk_tool", "SDK tool", {})
    async def sdk_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "from SDK"}
    
    sdk_server = create_sdk_mcp_server(
        name="sdk-server",
        tools=[sdk_tool]
    )
    
    # Create configuration with both SDK and external servers
    external_server = {
        "type": "stdio",
        "command": "echo",
        "args": ["test"]
    }
    
    options = ClaudeCodeOptions(
        mcp_servers={
            "sdk": sdk_server,
            "external": external_server
        }
    )
    
    # Verify both server types are in the configuration
    assert "sdk" in options.mcp_servers
    assert "external" in options.mcp_servers
    assert options.mcp_servers["sdk"]["type"] == "sdk"
    assert options.mcp_servers["external"]["type"] == "stdio"


@pytest.mark.asyncio
async def test_server_creation():
    """Test that SDK MCP servers are created correctly."""
    server = create_sdk_mcp_server(
        name="test-server",
        version="2.0.0",
        tools=[]
    )
    
    # Verify server configuration
    assert server["type"] == "sdk"
    assert server["name"] == "test-server"
    assert "instance" in server
    # The instance should be an MCP server (we can't check the exact type without importing mcp)
    assert server["instance"] is not None