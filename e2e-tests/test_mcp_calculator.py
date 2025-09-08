"""End-to-end tests for MCP calculator with real Claude API calls.

These tests verify that MCP tools are properly executed through the full stack,
including actual API calls to Claude.
"""

import math
from typing import Any

import pytest

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)
from claude_code_sdk.types import TextBlock, ToolResultBlock, ToolUseBlock


# Define calculator tools (reuse from examples)
@tool("add", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Add two numbers together."""
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}]}


@tool("subtract", "Subtract one number from another", {"a": float, "b": float})
async def subtract_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Subtract b from a."""
    result = args["a"] - args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} - {args['b']} = {result}"}]}


@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Multiply two numbers."""
    result = args["a"] * args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} × {args['b']} = {result}"}]}


@tool("divide", "Divide one number by another", {"a": float, "b": float})
async def divide_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Divide a by b."""
    if args["b"] == 0:
        return {
            "content": [{"type": "text", "text": "Error: Division by zero is not allowed"}],
            "is_error": True,
        }
    result = args["a"] / args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} ÷ {args['b']} = {result}"}]}


@tool("sqrt", "Calculate square root", {"n": float})
async def square_root(args: dict[str, Any]) -> dict[str, Any]:
    """Calculate the square root of a number."""
    n = args["n"]
    if n < 0:
        return {
            "content": [
                {"type": "text", "text": f"Error: Cannot calculate square root of negative number {n}"}
            ],
            "is_error": True,
        }
    result = math.sqrt(n)
    return {"content": [{"type": "text", "text": f"√{n} = {result}"}]}


@tool("power", "Raise a number to a power", {"base": float, "exponent": float})
async def power(args: dict[str, Any]) -> dict[str, Any]:
    """Raise base to the exponent power."""
    result = args["base"] ** args["exponent"]
    return {"content": [{"type": "text", "text": f"{args['base']}^{args['exponent']} = {result}"}]}


@pytest.fixture
def calculator_options():
    """Create ClaudeCodeOptions with calculator MCP server."""
    calculator = create_sdk_mcp_server(
        name="calculator",
        version="2.0.0",
        tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers, square_root, power],
    )
    
    return ClaudeCodeOptions(
        mcp_servers={"calc": calculator},
        allowed_tools=[
            "mcp__calc__add",
            "mcp__calc__subtract",
            "mcp__calc__multiply",
            "mcp__calc__divide",
            "mcp__calc__sqrt",
            "mcp__calc__power",
        ],
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_basic_addition(calculator_options):
    """Test that addition tool is properly called and returns correct result."""
    tool_calls_found = False
    result_found = False
    
    async with ClaudeSDKClient(options=calculator_options) as client:
        await client.query("Calculate 15 + 27")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "mcp__calc__add":
                            tool_calls_found = True
                            assert block.input == {"a": 15, "b": 27}
                    elif isinstance(block, TextBlock):
                        if "42" in block.text:
                            result_found = True
    
    assert tool_calls_found, "Add tool was not called"
    assert result_found, "Result 42 not found in response"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_division(calculator_options):
    """Test division tool with decimal result."""
    tool_calls_found = False
    result_text = None
    
    async with ClaudeSDKClient(options=calculator_options) as client:
        await client.query("What is 100 divided by 7?")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "mcp__calc__divide":
                            tool_calls_found = True
                            assert block.input == {"a": 100, "b": 7}
                    elif isinstance(block, TextBlock):
                        result_text = block.text
    
    assert tool_calls_found, "Divide tool was not called"
    assert result_text and "14.2857" in result_text, f"Expected decimal result in: {result_text}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_square_root(calculator_options):
    """Test square root calculation."""
    tool_calls_found = False
    result_found = False
    
    async with ClaudeSDKClient(options=calculator_options) as client:
        await client.query("Calculate the square root of 144")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "mcp__calc__sqrt":
                            tool_calls_found = True
                            assert block.input == {"n": 144}
                    elif isinstance(block, TextBlock):
                        if "12" in block.text:
                            result_found = True
    
    assert tool_calls_found, "Square root tool was not called"
    assert result_found, "Result 12 not found in response"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_power(calculator_options):
    """Test raising numbers to powers."""
    tool_calls_found = False
    result_found = False
    
    async with ClaudeSDKClient(options=calculator_options) as client:
        await client.query("What is 2 raised to the power of 8?")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "mcp__calc__power":
                            tool_calls_found = True
                            assert block.input == {"base": 2, "exponent": 8}
                    elif isinstance(block, TextBlock):
                        if "256" in block.text:
                            result_found = True
    
    assert tool_calls_found, "Power tool was not called"
    assert result_found, "Result 256 not found in response"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_step_calculation(calculator_options):
    """Test that multiple tools can be used in sequence for complex calculations."""
    tool_calls = []
    
    async with ClaudeSDKClient(options=calculator_options) as client:
        await client.query("Calculate (12 + 8) * 3 - 10")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_calls.append(block.name)
    
    # Should use multiple calculator tools (add, multiply, subtract)
    assert len(tool_calls) >= 2, f"Expected multiple tool calls, got: {tool_calls}"
    # Verify calculator tools were used
    calc_tools_used = [t for t in tool_calls if t.startswith("mcp__calc__")]
    assert len(calc_tools_used) >= 2, f"Expected at least 2 calculator tools, got: {calc_tools_used}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tool_permissions_enforced():
    """Test that disallowed_tools prevents tool execution."""
    calculator = create_sdk_mcp_server(
        name="calculator",
        version="2.0.0",
        tools=[add_numbers],
    )
    
    # Use disallowed_tools to explicitly block the calculator tool
    options = ClaudeCodeOptions(
        mcp_servers={"calc": calculator},
        disallowed_tools=["mcp__calc__add"]
    )
    
    add_tool_called = False
    
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Calculate 5 + 3")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "mcp__calc__add":
                            add_tool_called = True
    
    # Explicitly disallowed tool should NOT be called
    assert not add_tool_called, "mcp__calc__add was called despite being in disallowed_tools"