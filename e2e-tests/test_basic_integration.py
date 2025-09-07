#!/usr/bin/env python3
"""Basic integration test for Claude Code SDK."""

import anyio

from claude_code_sdk import AssistantMessage, TextBlock, query


async def test_basic_query():
    """Test basic query functionality."""
    print("Testing basic query...")
    found_response = False
    
    async for message in query(prompt="What is 2 + 2?"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Response: {block.text}")
                    found_response = True
    
    if not found_response:
        raise Exception("No response received from Claude")
    
    print("✅ Basic query test passed")


async def test_with_system_prompt():
    """Test query with custom system prompt."""
    print("\nTesting with system prompt...")
    found_response = False
    
    from claude_code_sdk import ClaudeCodeOptions
    
    options = ClaudeCodeOptions(
        system_prompt="You are a helpful assistant that answers concisely.",
        max_turns=1,
    )
    
    async for message in query(
        prompt="What is Python in one sentence?", options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Response: {block.text}")
                    found_response = True
    
    if not found_response:
        raise Exception("No response received from Claude")
    
    print("✅ System prompt test passed")


async def main():
    """Run all integration tests."""
    print("=" * 50)
    print("Running Claude Code SDK Integration Tests")
    print("=" * 50)
    
    try:
        await test_basic_query()
        await test_with_system_prompt()
        
        print("\n" + "=" * 50)
        print("✅ All integration tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    anyio.run(main())