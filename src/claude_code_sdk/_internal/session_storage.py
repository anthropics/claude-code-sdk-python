"""Session storage and persistence utilities."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import ClaudeCodeOptions, Message


@dataclass
class SessionData:
    """Session data for persistence."""
    
    session_id: str
    start_time: datetime
    last_activity: datetime
    conversation_history: list[Message] = field(default_factory=list)
    working_directory: str = ""
    options: ClaudeCodeOptions | None = None
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
        self.last_activity = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        from ..types import UserMessage, AssistantMessage, SystemMessage, ResultMessage
        
        # Convert messages to serializable format
        serialized_messages = []
        for msg in self.conversation_history:
            msg_dict = {
                "message_type": type(msg).__name__,
            }
            
            if isinstance(msg, UserMessage):
                msg_dict.update({
                    "content": msg.content,
                })
            elif isinstance(msg, AssistantMessage):
                msg_dict.update({
                    "content": [
                        {
                            "type": type(block).__name__,
                            "text": getattr(block, 'text', None),
                            "id": getattr(block, 'id', None),
                            "name": getattr(block, 'name', None),
                            "input": getattr(block, 'input', None),
                            "tool_use_id": getattr(block, 'tool_use_id', None),
                            "is_error": getattr(block, 'is_error', None),
                        }
                        for block in msg.content
                    ],
                })
            elif isinstance(msg, SystemMessage):
                msg_dict.update({
                    "subtype": msg.subtype,
                    "data": msg.data,
                })
            elif isinstance(msg, ResultMessage):
                msg_dict.update({
                    "subtype": msg.subtype,
                    "duration_ms": msg.duration_ms,
                    "duration_api_ms": msg.duration_api_ms,
                    "is_error": msg.is_error,
                    "num_turns": msg.num_turns,
                    "session_id": msg.session_id,
                    "total_cost_usd": msg.total_cost_usd,
                    "usage": msg.usage,
                    "result": msg.result,
                })
            
            serialized_messages.append(msg_dict)
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": serialized_messages,
            "working_directory": self.working_directory,
            "options": {
                "model": self.options.model if self.options else None,
                "allowed_tools": self.options.allowed_tools if self.options else [],
                "permission_mode": self.options.permission_mode if self.options else None,
            } if self.options else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary (JSON deserialization)."""
        from ..types import UserMessage, AssistantMessage, SystemMessage, ResultMessage, TextBlock, ToolUseBlock, ToolResultBlock
        
        # Parse conversation history
        conversation_history = []
        for msg_data in data.get("conversation_history", []):
            message_type = msg_data.get("message_type", "")
            
            if message_type == "UserMessage":
                message = UserMessage(content=msg_data["content"])
            elif message_type == "AssistantMessage":
                # Reconstruct content blocks
                content_blocks = []
                for block_data in msg_data.get("content", []):
                    block_type = block_data.get("type", "")
                    if block_type == "TextBlock":
                        content_blocks.append(TextBlock(text=block_data.get("text", "")))
                    elif block_type == "ToolUseBlock":
                        content_blocks.append(ToolUseBlock(
                            id=block_data.get("id", ""),
                            name=block_data.get("name", ""),
                            input=block_data.get("input", {})
                        ))
                    elif block_type == "ToolResultBlock":
                        content_blocks.append(ToolResultBlock(
                            tool_use_id=block_data.get("tool_use_id", ""),
                            content=block_data.get("content"),
                            is_error=block_data.get("is_error")
                        ))
                message = AssistantMessage(content=content_blocks)
            elif message_type == "SystemMessage":
                message = SystemMessage(
                    subtype=msg_data.get("subtype", ""),
                    data=msg_data.get("data", {})
                )
            elif message_type == "ResultMessage":
                message = ResultMessage(
                    subtype=msg_data.get("subtype", ""),
                    duration_ms=msg_data.get("duration_ms", 0),
                    duration_api_ms=msg_data.get("duration_api_ms", 0),
                    is_error=msg_data.get("is_error", False),
                    num_turns=msg_data.get("num_turns", 0),
                    session_id=msg_data.get("session_id", ""),
                    total_cost_usd=msg_data.get("total_cost_usd"),
                    usage=msg_data.get("usage"),
                    result=msg_data.get("result")
                )
            else:
                continue  # Skip unknown message types
            
            conversation_history.append(message)
        
        # Parse options
        options = None
        if data.get("options"):
            options = ClaudeCodeOptions(
                model=data["options"].get("model"),
                allowed_tools=data["options"].get("allowed_tools", []),
                permission_mode=data["options"].get("permission_mode"),
            )
        
        return cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            conversation_history=conversation_history,
            working_directory=data.get("working_directory", ""),
            options=options,
        )


class SimpleSessionPersistence:
    """Simple file-based session persistence."""
    
    def __init__(self, storage_path: Path | str | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".claude_sdk" / "sessions"
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    async def save_session(self, session_data: SessionData) -> None:
        """Save session data to file."""
        file_path = self._storage_path / f"{session_data.session_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(session_data.to_dict(), f, indent=2, ensure_ascii=False)
    
    async def load_session(self, session_id: str) -> SessionData | None:
        """Load session data from file."""
        file_path = self._storage_path / f"{session_id}.json"
        if not file_path.exists():
            return None
        
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return SessionData.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    async def list_sessions(self) -> list[str]:
        """List all session IDs."""
        session_ids = []
        for file_path in self._storage_path.glob("*.json"):
            session_ids.append(file_path.stem)  # filename without .json extension
        return sorted(session_ids)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session file."""
        file_path = self._storage_path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False