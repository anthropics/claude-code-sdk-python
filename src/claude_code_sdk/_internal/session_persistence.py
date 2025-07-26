"""Session data management and local file persistence."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import ClaudeCodeOptions

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Individual message in conversation history."""

    role: str
    content: str
    timestamp: datetime
    message_type: str = "text"
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_type=data.get("message_type", "text"),
            usage=data.get("usage"),
        )


@dataclass
class SessionData:
    """Complete session data including history and metadata."""

    session_id: str
    start_time: datetime
    last_activity: datetime
    conversation_history: list[ConversationMessage] = field(default_factory=list)
    options: ClaudeCodeOptions | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    working_directory: str | None = None

    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.conversation_history)

    def add_message(self, message: ConversationMessage) -> None:
        """Add message to conversation history."""
        self.conversation_history.append(message)
        self.last_activity = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": [
                msg.to_dict() for msg in self.conversation_history
            ],
            "options": self.options.__dict__ if self.options else None,
            "metadata": self.metadata,
            "working_directory": self.working_directory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        options = None
        if data.get("options"):
            options = ClaudeCodeOptions(**data["options"])

        return cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            conversation_history=[
                ConversationMessage.from_dict(msg)
                for msg in data.get("conversation_history", [])
            ],
            options=options,
            metadata=data.get("metadata", {}),
            working_directory=data.get("working_directory"),
        )


@dataclass
class SessionSummary:
    """Summary information for session listing."""

    session_id: str
    start_time: datetime
    last_activity: datetime
    message_count: int
    first_message_preview: str | None = None


class SessionPersistence:
    """Manages session data persistence using JSON files."""

    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize session persistence.

        Args:
            storage_path: Directory for session files (default: ~/.claude_sdk/sessions/)
        """
        if storage_path is None:
            storage_path = Path.home() / ".claude_sdk" / "sessions"

        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def _get_session_file_path(self, session_id: str) -> Path:
        """Get file path for session."""
        return self._storage_path / f"{session_id}.json"

    async def save_session(self, session_data: SessionData) -> None:
        """
        Save session data to JSON file.

        Args:
            session_data: Session data to save
        """
        file_path = self._get_session_file_path(session_data.session_id)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session_data.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved session {session_data.session_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save session {session_data.session_id}: {e}")
            raise

    async def load_session(self, session_id: str) -> SessionData | None:
        """
        Load session data from JSON file.

        Args:
            session_id: Session ID to load

        Returns:
            SessionData if found, None otherwise
        """
        file_path = self._get_session_file_path(session_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return SessionData.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session file.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_session_file_path(session_id)

        if file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Deleted session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
                raise
        return False

    async def list_sessions(
        self, limit: int = 50, order_by: str = "last_activity"
    ) -> list[SessionSummary]:
        """
        List available sessions.

        Args:
            limit: Maximum sessions to return
            order_by: Sort field ('start_time', 'last_activity', 'message_count')

        Returns:
            List of session summaries
        """
        sessions = []

        for file_path in self._storage_path.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Extract first user message for preview
                first_message_preview = None
                for msg in data.get("conversation_history", []):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        first_message_preview = (
                            content[:100] + "..." if len(content) > 100 else content
                        )
                        break

                sessions.append(
                    SessionSummary(
                        session_id=data["session_id"],
                        start_time=datetime.fromisoformat(data["start_time"]),
                        last_activity=datetime.fromisoformat(data["last_activity"]),
                        message_count=len(data.get("conversation_history", [])),
                        first_message_preview=first_message_preview,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to read session file {file_path}: {e}")
                continue

        # Sort sessions
        if order_by == "start_time":
            sessions.sort(key=lambda s: s.start_time, reverse=True)
        elif order_by == "message_count":
            sessions.sort(key=lambda s: s.message_count, reverse=True)
        else:  # default to last_activity
            sessions.sort(key=lambda s: s.last_activity, reverse=True)

        return sessions[:limit]

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self._get_session_file_path(session_id).exists()

    async def export_session(
        self, session_id: str, export_path: Path | str, format: str = "json"
    ) -> None:
        """
        Export session to different format.

        Args:
            session_id: Session to export
            export_path: Output file path
            format: Export format ('json', 'markdown', 'txt')
        """
        session_data = await self.load_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")

        export_path = Path(export_path)

        if format == "json":
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(session_data.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "markdown":
            await self._export_markdown(session_data, export_path)
        elif format == "txt":
            await self._export_text(session_data, export_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _export_markdown(
        self, session_data: SessionData, export_path: Path
    ) -> None:
        """Export session as markdown."""
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"# Claude Conversation - {session_data.session_id}\n\n")
            f.write(
                f"**Started:** {session_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                f"**Last Activity:** {session_data.last_activity.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"**Messages:** {session_data.message_count}\n\n")

            for msg in session_data.conversation_history:
                role_display = "**User**" if msg.role == "user" else "**Claude**"
                f.write(f"## {role_display} ({msg.timestamp.strftime('%H:%M:%S')})\n\n")
                f.write(f"{msg.content}\n\n")

    async def _export_text(self, session_data: SessionData, export_path: Path) -> None:
        """Export session as plain text."""
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"Claude Conversation - {session_data.session_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Started: {session_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                f"Last Activity: {session_data.last_activity.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Messages: {session_data.message_count}\n\n")

            for msg in session_data.conversation_history:
                role_display = "User" if msg.role == "user" else "Claude"
                f.write(f"[{msg.timestamp.strftime('%H:%M:%S')}] {role_display}:\n")
                f.write(f"{msg.content}\n\n")
