#!/usr/bin/env python3
"""Example: Calculator MCP Server.

This example demonstrates how to create an in-process MCP server with
calculator tools using the Claude Code Python SDK.

Unlike external MCP servers that require separate processes, this server
runs directly within your Python application, providing better performance
and simpler deployment.
"""

import asyncio
from typing import Any, Dict

from claude_code_sdk import (
    ClaudeCodeOptions,
    create_sdk_mcp_server,
    query,
    tool,
)


# Define calculator tools using the @tool decorator

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add two numbers together."""
    result = args["a"] + args["b"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"{args['a']} + {args['b']} = {result}"
            }
        ]
    }


@tool("subtract", "Subtract one number from another", {"a": float, "b": float})
async def subtract_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    """Subtract b from a."""
    result = args["a"] - args["b"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"{args['a']} - {args['b']} = {result}"
            }
        ]
    }


@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    """Multiply two numbers together."""
    result = args["a"] * args["b"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"{args['a']} × {args['b']} = {result}"
            }
        ]
    }


@tool("divide", "Divide one number by another", {"a": float, "b": float})
async def divide_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    """Divide a by b."""
    if args["b"] == 0:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Error: Division by zero is not allowed"
                }
            ],
            "is_error": True
        }
    
    result = args["a"] / args["b"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"{args['a']} ÷ {args['b']} = {result}"
            }
        ]
    }


@tool("sqrt", "Calculate square root", {"n": float})
async def square_root(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the square root of a number."""
    n = args["n"]
    if n < 0:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: Cannot calculate square root of negative number {n}"
                }
            ],
            "is_error": True
        }
    
    result = n ** 0.5
    return {
        "content": [
            {
                "type": "text",
                "text": f"√{n} = {result}"
            }
        ]
    }


async def main():
    """Run the calculator example."""
    # Create the in-process MCP server with all calculator tools
    calculator_server = create_sdk_mcp_server(
        name="calculator",
        version="1.0.0",
        tools=[
            add_numbers,
            subtract_numbers,
            multiply_numbers,
            divide_numbers,
            square_root
        ]
    )
    
    # Configure Claude SDK to use our calculator server
    options = ClaudeCodeOptions(
        mcp_servers={"calculator": calculator_server},
        # Allow the calculator tools to be used
        allowed_tools=["add", "subtract", "multiply", "divide", "sqrt"],
        # Limit to one turn for this example
        max_turns=1
    )
    
    # Example calculations to perform
    prompt = """
    Please help me with these calculations:
    1. What is 15 + 27?
    2. What is 100 - 45?
    3. What is 12 × 8?
    4. What is 144 ÷ 12?
    5. What is the square root of 256?
    """
    
    print("🧮 Calculator MCP Server Example")
    print("=" * 50)
    print(f"Prompt: {prompt.strip()}")
    print("=" * 50)
    print("\nClaude's response:")
    print("-" * 50)
    
    # Query Claude with our calculator server
    async for message in query(prompt=prompt, options=options):
        # Print assistant messages
        if hasattr(message, 'content'):
            for block in message.content:
                if hasattr(block, 'text'):
                    print(block.text)
    
    print("-" * 50)
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())