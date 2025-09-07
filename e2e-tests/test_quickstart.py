#!/usr/bin/env python3
"""E2E test based on quickstart examples."""

import anyio

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)


async def test_basic_example():
    """Test basic example - simple question."""
    print("Testing basic example...")
    
    found_response = False
    async for message in query(prompt="What is 2 + 2?"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"  Response: {block.text}")
                    found_response = True
    
    if not found_response:
        raise Exception("No response received for basic example")
    print("✅ Basic example test passed")


async def test_with_options_example():
    """Test example with custom options."""
    print("\nTesting with options example...")
    
    options = ClaudeCodeOptions(
        system_prompt="You are a helpful assistant that explains things simply.",
        max_turns=1,
    )
    
    found_response = False
    async for message in query(
        prompt="Explain what Python is in one sentence.", options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"  Response: {block.text}")
                    found_response = True
    
    if not found_response:
        raise Exception("No response received for options example")
    print("✅ Options example test passed")


async def test_with_tools_example():
    """Test example using tools."""
    print("\nTesting with tools example...")
    
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write"],
        system_prompt="You are a helpful file assistant.",
    )
    
    found_response = False
    found_cost = False
    async for message in query(
        prompt="Create a file called hello.txt with 'Hello, World!' in it",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"  Response: {block.text[:100]}...")
                    found_response = True
        elif isinstance(message, ResultMessage) and message.total_cost_usd > 0:
            print(f"  Cost: ${message.total_cost_usd:.4f}")
            found_cost = True
    
    if not found_response:
        raise Exception("No response received for tools example")
    if not found_cost:
        print("  Note: Cost information not available (might be expected)")
    print("✅ Tools example test passed")


async def main():
    """Run all quickstart tests."""
    print("=" * 50)
    print("Running Quickstart E2E Tests")
    print("=" * 50)
    
    try:
        await test_basic_example()
        await test_with_options_example()
        await test_with_tools_example()
        
        print("\n" + "=" * 50)
        print("✅ All quickstart tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    anyio.run(main)