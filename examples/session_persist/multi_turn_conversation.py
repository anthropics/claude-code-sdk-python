#!/usr/bin/env python3
"""
Multi-turn conversation with session resumption example using SessionPersistentClient.

This demonstrates how the SessionPersistentClient automatically captures
session context across multiple turns of conversation, and how to resume
sessions after disconnecting. The demo shows:

1. Initial conversation (Turns 1-2) with automatic session persistence
2. Disconnect from the session 
3. Resume the session using start_or_resume_session()
4. Continue conversation (Turns 3-4) with preserved context

Key features demonstrated:
- Local session data loading from storage
- CLI --resume option for server-side session continuity  
- Context retention across disconnect/resume cycles
- Seamless multi-turn conversation flow
"""

import trio
from pathlib import Path

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    SessionPersistentClient,
    TextBlock,
)


async def multi_turn_demo():
    """Demonstrate multi-turn conversation with session resumption."""
    print("=== Multi-Turn Conversation with Session Resumption ===")
    
    storage_path = Path("./conversation_sessions")
    
    # Phase 1: Initial conversation (Turns 1-2)
    print("\n🚀 Phase 1: Initial Conversation")
    session_id = None
    
    async with SessionPersistentClient(
        options=ClaudeCodeOptions(),
        storage_path=storage_path
    ) as client:
        
        # Turn 1: Introduction
        print("\n💬 Turn 1: Introduction")
        intro_msg = "Hello! My name is Alex and I'm a software developer."
        await client.query(intro_msg)
        print(f"User: {intro_msg}")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
        
        # Turn 2: Ask about a topic
        print("\n💬 Turn 2: Technical question")
        await client.query("What's the difference between async and sync programming?")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
        
        # Capture session ID for resumption
        session_id = client.get_current_session_id()
        session_data = await client.load_session(session_id)
        
        print(f"\n📋 Session after 2 turns:")
        print(f"   ID: {session_id}")
        print(f"   Total Messages: {len(session_data.conversation_history) if session_data else 'Unknown'}")
        print(f"   🔌 Disconnecting to demonstrate session resumption...")
    
    # Phase 2: Resume session (Turns 3-4)
    print(f"\n🔄 Phase 2: Resuming Session {session_id}")
    
    # Create new client instance and resume the session
    client = SessionPersistentClient(
        options=ClaudeCodeOptions(),
        storage_path=storage_path
    )
    
    try:
        # Resume the session - this loads local data AND configures CLI --resume
        await client.start_or_resume_session(session_id)
        print(f"✅ Session resumed. Local data: {len(client._session_data.conversation_history) if client._session_data else 0} messages")
        
        # Connect and continue the conversation
        await client.connect()
        
        # Turn 3: Follow-up question (tests context retention across disconnect/resume)
        print("\n💬 Turn 3: Follow-up question")
        await client.query("Can you give me a Python example of what you just explained?")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
        
        # Turn 4: Reference earlier context
        print("\n💬 Turn 4: Reference earlier context")
        await client.query("Thanks! What was my name again that I mentioned at the beginning?")
        
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
        
        # Show final session info
        final_session_id = client.get_current_session_id()
        session_data = await client.load_session(final_session_id)
        
        print(f"\n📋 Final Session Summary:")
        print(f"   Original ID: {session_id}")
        print(f"   Current ID: {final_session_id}")
        print(f"   Total Messages: {len(session_data.conversation_history) if session_data else 'Unknown'}")
        print(f"   Duration: {session_data.last_activity - session_data.start_time if session_data else 'Unknown'}")
        print(f"   Auto-saved to: {storage_path.absolute()}")
        
    finally:
        await client.disconnect()


async def main():
    """Run the multi-turn conversation with session resumption demo."""
    await multi_turn_demo()
    
    print("\n" + "="*60)
    print("✅ Session resumption demo complete!")
    print("📝 Key features demonstrated:")
    print("  • Local session data is loaded from storage")
    print("  • CLI --resume option is set for server-side continuity")
    print("  • Conversation context is maintained across disconnect/resume")
    print("  • Multi-turn conversations work seamlessly across sessions")


if __name__ == "__main__":
    trio.run(main)