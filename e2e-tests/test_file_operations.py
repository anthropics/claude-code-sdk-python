#!/usr/bin/env python3
"""Test file operations with Claude Code SDK."""

import os
import tempfile
import anyio
from pathlib import Path

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, TextBlock, query


async def test_file_creation():
    """Test file creation through SDK."""
    print("Testing file creation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_output.txt"
        
        options = ClaudeCodeOptions(
            allowed_tools=["Write"],
            system_prompt="You are a helpful file assistant. Be concise.",
            max_turns=1,
        )
        
        prompt = f"Create a file at {test_file} with content 'Hello from Claude Code SDK!'"
        
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text[:100]}...")  # Print first 100 chars
        
        # Verify file was created
        if not test_file.exists():
            raise Exception(f"File {test_file} was not created")
        
        content = test_file.read_text()
        if "Hello from Claude Code SDK!" not in content:
            raise Exception(f"File content incorrect: {content}")
        
        print(f"✅ File created successfully with correct content")


async def test_file_reading():
    """Test file reading through SDK."""
    print("\nTesting file reading...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "input.txt"
        test_content = "This is a test file for Claude Code SDK e2e testing."
        test_file.write_text(test_content)
        
        options = ClaudeCodeOptions(
            allowed_tools=["Read"],
            system_prompt="You are a helpful file assistant. Be concise.",
            max_turns=1,
        )
        
        prompt = f"Read the file at {test_file} and tell me the first word in the file"
        
        found_response = False
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text[:100]}...")
                        # Check if Claude identified "This" as the first word
                        if "This" in block.text or "this" in block.text.lower():
                            found_response = True
        
        if not found_response:
            raise Exception("Claude did not identify the first word correctly")
        
        print("✅ File reading test passed")


async def main():
    """Run file operation tests."""
    print("=" * 50)
    print("Running File Operation Tests")
    print("=" * 50)
    
    try:
        await test_file_creation()
        await test_file_reading()
        
        print("\n" + "=" * 50)
        print("✅ All file operation tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ File operation test failed: {e}")
        raise


if __name__ == "__main__":
    anyio.run(main())