# Session Persistence Examples

This directory contains examples demonstrating the `SessionPersistentClient` that provides automatic session persistence by wrapping `ClaudeSDKClient`.

## Design Philosophy

The `SessionPersistentClient` follows a clean wrapper pattern:

1. **Wraps ClaudeSDKClient**: Does not modify the core client, just wraps it
2. **Uses receive_messages()**: Extracts session data from actual message responses
3. **Server-Generated IDs**: Uses session IDs from Claude's server, not client-generated UUIDs
4. **Automatic Persistence**: Saves conversation data transparently in the background

## Examples

### 1. `simple_persist.py`
**Purpose**: Basic demonstration of automatic session persistence

**Key Features**:
- Shows how SessionPersistentClient wraps ClaudeSDKClient
- Demonstrates automatic session ID extraction from messages
- Shows session data inspection capabilities
- Simple conversation with automatic saving

**Run it**:
```bash
python examples/session_persist/simple_persist.py
```

### 2. `multi_turn_conversation.py`
**Purpose**: Multi-turn conversation with session resumption demonstration

**Key Features**:
- Multi-turn conversation that maintains context
- Session disconnect and resumption workflow
- Resume session using `start_or_resume_session()`
- Shows both local data loading and CLI --resume functionality
- Demonstrates conversation context continuity across disconnect/resume
- Context reference across multiple turns spanning session boundaries

**Demo Flow**:
1. **Phase 1**: Initial conversation (Turns 1-2) with automatic persistence
2. **Disconnect**: Session is saved and connection closed
3. **Phase 2**: Resume session with new client instance 
4. **Continue**: Turns 3-4 with preserved context from earlier turns

**Run it**:
```bash
python examples/session_persist/multi_turn_conversation.py
```

## How SessionPersistentClient Works

### Architecture

```python
SessionPersistentClient
├── ClaudeSDKClient (wrapped)          # Handles all Claude interactions
├── SessionPersistence                 # Manages file storage
└── Message Processing                 # Extracts session data from messages
```

### Key Methods

```python
# Initialize with automatic persistence
client = SessionPersistentClient(
    options=ClaudeCodeOptions(),
    storage_path="./my_sessions"
)

# All ClaudeSDKClient methods are available:
await client.connect()
await client.query("Hello")
async for message in client.receive_response():
    # Session data is automatically extracted and saved
    print(message)

# Session management:
await client.start_or_resume_session(id)      # Resume existing session (local + server)
session_id = client.get_current_session_id()  # Server-generated ID
sessions = await client.list_sessions()       # List all saved sessions
session_data = await client.load_session(id)  # Load session for inspection
await client.delete_session(session_id)       # Delete a session
```

### Automatic Session Extraction

The client automatically extracts session data from messages:

1. **Session ID Detection**: Looks for `session_id` in message metadata
2. **Message Conversion**: Converts Claude messages to `ConversationMessage` format
3. **Automatic Saving**: Saves session data after each message
4. **Context Preservation**: Maintains conversation history and metadata

### File Structure

Sessions are saved as JSON files:
```
~/.claude_sdk/sessions/
├── 2aecab00-6512-4e29-9da3-9321cac6eb2.json
├── 7b8f3c45-2d19-4e7a-b6c1-f9d2e8a3c7b5.json
└── ...
```

Each file contains:
```json
{
  "session_id": "server-generated-uuid",
  "start_time": "2025-07-30T15:25:31.069594",
  "last_activity": "2025-07-30T15:27:45.123456",
  "conversation_history": [...],
  "working_directory": "/path/to/working/dir",
  "options": {...}
}
```

## Benefits of This Design

### ✅ Clean Separation
- Core `ClaudeSDKClient` remains unchanged
- Persistence is an optional wrapper layer
- No mixing of concerns

### ✅ Server-Driven
- Uses actual session IDs from Claude's server
- No client-side UUID generation
- Matches Claude's internal session management

### ✅ Automatic Operation
- No manual session management required
- Transparent persistence in background
- Works with all `ClaudeSDKClient` features

### ✅ Easy Migration
- Existing `ClaudeSDKClient` code works unchanged
- Just replace `ClaudeSDKClient` with `SessionPersistentClient`
- All methods and features preserved

## Usage Pattern

### Old (no persistence):
```python
async with ClaudeSDKClient() as client:
    await client.query("Hello")
    async for message in client.receive_response():
        print(message)
```

### New (with automatic persistence):
```python
async with SessionPersistentClient() as client:
    await client.query("Hello")
    async for message in client.receive_response():
        print(message)  # Same code, automatic persistence!
```

## Dependencies

These examples use `trio` for async operations. Install with:
```bash
pip install trio
```

Or use standard `asyncio` by replacing `trio.run()` with `asyncio.run()`.