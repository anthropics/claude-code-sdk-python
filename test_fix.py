#!/usr/bin/env python3
"""Test script to verify the cost_usd fix works."""

import anyio
from claude_code_sdk import process_query, ClaudeCodeOptions

async def main():
    """Test the SDK with a simple query."""
    print("Testing claude-code-sdk with cost_usd fix...")
    
    try:
        # Simple test query
        async for message in process_query(
            "What is 2+2?",
            ClaudeCodeOptions()
        ):
            print(f"Message type: {type(message).__name__}")
            print(f"Message: {message}")
            print("-" * 40)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    anyio.run(main)