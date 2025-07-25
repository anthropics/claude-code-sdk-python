"""Slash command system for StatefulCLIClient."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..stateful_client import StatefulCLIClient

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of executing a slash command."""
    command: str
    success: bool
    result: Any
    message: str | None = None


class SlashCommand(ABC):
    """Base class for slash commands."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Execute the command."""
        pass


class HelpCommand(SlashCommand):
    """Show available commands."""
    
    def __init__(self):
        super().__init__("help", "Show available commands")
    
    async def execute(self, args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Show help for all registered commands."""
        help_text = "Available slash commands:\n\n"
        
        for cmd_name, cmd in client._command_router._commands.items():
            help_text += f"/{cmd_name} - {cmd.description}\n"
        
        return CommandResult(
            command="help",
            success=True,
            result=help_text,
            message="Command help displayed"
        )


class ExitCommand(SlashCommand):
    """Exit the current session."""
    
    def __init__(self):
        super().__init__("exit", "End session gracefully")
    
    async def execute(self, _args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """End the current session."""
        await client.end_session()
        
        return CommandResult(
            command="exit",
            success=True,
            result="Session ended",
            message="Goodbye!"
        )


class ClearCommand(SlashCommand):
    """Clear conversation history."""
    
    def __init__(self):
        super().__init__("clear", "Clear conversation history")
    
    async def execute(self, _args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Clear the conversation history."""
        if client._transport and client._transport.get_session_data():
            session_data = client._transport.get_session_data()
            message_count = len(session_data.conversation_history)
            session_data.conversation_history.clear()
            await client._transport.save_session()
            
            return CommandResult(
                command="clear",
                success=True,
                result=f"Cleared {message_count} messages",
                message="Conversation history cleared"
            )
        
        return CommandResult(
            command="clear",
            success=False,
            result=None,
            message="No active session to clear"
        )


class HistoryCommand(SlashCommand):
    """Show conversation history."""
    
    def __init__(self):
        super().__init__("history", "Show recent messages [limit]")
    
    async def execute(self, args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Show conversation history."""
        limit = 10  # default
        
        if args and args[0].isdigit():
            limit = int(args[0])
        
        if client._transport and client._transport.get_session_data():
            session_data = client._transport.get_session_data()
            recent_messages = session_data.conversation_history[-limit:]
            
            history_text = f"Recent {len(recent_messages)} messages:\n\n"
            for msg in recent_messages:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                role = msg.role.capitalize()
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                history_text += f"[{timestamp}] {role}: {preview}\n\n"
            
            return CommandResult(
                command="history",
                success=True,
                result=history_text,
                message=f"Showing {len(recent_messages)} recent messages"
            )
        
        return CommandResult(
            command="history",
            success=False,
            result=None,
            message="No active session"
        )


class SessionsCommand(SlashCommand):
    """List all sessions."""
    
    def __init__(self):
        super().__init__("sessions", "List all sessions")
    
    async def execute(self, _args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """List available sessions."""
        if client._persistence:
            sessions = await client._persistence.list_sessions(limit=20)
            
            if not sessions:
                return CommandResult(
                    command="sessions",
                    success=True,
                    result="No sessions found",
                    message="No sessions available"
                )
            
            sessions_text = "Available sessions:\n\n"
            for session in sessions:
                active_marker = " (ACTIVE)" if session.session_id == client.get_session_id() else ""
                start_time = session.start_time.strftime("%Y-%m-%d %H:%M")
                sessions_text += f"• {session.session_id}{active_marker}\n"
                sessions_text += f"  Started: {start_time}, Messages: {session.message_count}\n"
                if session.first_message_preview:
                    sessions_text += f"  Preview: {session.first_message_preview}\n"
                sessions_text += "\n"
            
            return CommandResult(
                command="sessions",
                success=True,
                result=sessions_text,
                message=f"Found {len(sessions)} sessions"
            )
        
        return CommandResult(
            command="sessions",
            success=False,
            result=None,
            message="Session persistence not available"
        )


class ResumeCommand(SlashCommand):
    """Resume a different session."""
    
    def __init__(self):
        super().__init__("resume", "Switch to different session <session_id>")
    
    async def execute(self, args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Resume a different session."""
        if not args:
            return CommandResult(
                command="resume",
                success=False,
                result=None,
                message="Please provide a session ID to resume"
            )
        
        session_id = args[0]
        
        try:
            success = await client.resume_session(session_id)
            if success:
                return CommandResult(
                    command="resume",
                    success=True,
                    result=f"Resumed session {session_id}",
                    message=f"Now active in session {session_id}"
                )
            else:
                return CommandResult(
                    command="resume",
                    success=False,
                    result=None,
                    message=f"Session {session_id} not found"
                )
        except Exception as e:
            return CommandResult(
                command="resume",
                success=False,
                result=None,
                message=f"Failed to resume session: {e}"
            )


class InfoCommand(SlashCommand):
    """Show session information."""
    
    def __init__(self):
        super().__init__("info", "Show session information")
    
    async def execute(self, _args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Show current session information."""
        if client._transport and client._transport.get_session_data():
            session_data = client._transport.get_session_data()
            
            info_text = f"Session Information:\n\n"
            info_text += f"Session ID: {session_data.session_id}\n"
            info_text += f"Started: {session_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            info_text += f"Last Activity: {session_data.last_activity.strftime('%Y-%m-%d %H:%M:%S')}\n"
            info_text += f"Messages: {session_data.message_count}\n"
            if session_data.working_directory:
                info_text += f"Working Directory: {session_data.working_directory}\n"
            if session_data.options and session_data.options.model:
                info_text += f"Model: {session_data.options.model}\n"
            
            return CommandResult(
                command="info",
                success=True,
                result=info_text,
                message="Session information displayed"
            )
        
        return CommandResult(
            command="info",
            success=False,
            result=None,
            message="No active session"
        )


class SearchCommand(SlashCommand):
    """Search conversation history."""
    
    def __init__(self):
        super().__init__("search", "Search conversation history <query>")
    
    async def execute(self, args: list[str], client: "StatefulCLIClient") -> CommandResult:
        """Search conversation history."""
        if not args:
            return CommandResult(
                command="search",
                success=False,
                result=None,
                message="Please provide a search query"
            )
        
        query = " ".join(args).lower()
        
        if client._transport and client._transport.get_session_data():
            session_data = client._transport.get_session_data()
            matching_messages = []
            
            for msg in session_data.conversation_history:
                if query in msg.content.lower():
                    matching_messages.append(msg)
            
            if not matching_messages:
                return CommandResult(
                    command="search",
                    success=True,
                    result=f"No messages found containing '{query}'",
                    message="No matches found"
                )
            
            results_text = f"Found {len(matching_messages)} messages containing '{query}':\n\n"
            for msg in matching_messages[-10:]:  # Show last 10 matches
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                role = msg.role.capitalize()
                # Highlight the search term (simple approach)
                content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                results_text += f"[{timestamp}] {role}: {content_preview}\n\n"
            
            return CommandResult(
                command="search",
                success=True,
                result=results_text,
                message=f"Found {len(matching_messages)} matching messages"
            )
        
        return CommandResult(
            command="search",
            success=False,
            result=None,
            message="No active session to search"
        )


class SlashCommandRouter:
    """Routes and executes slash commands."""
    
    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}
        self._register_builtin_commands()
    
    def _register_builtin_commands(self):
        """Register built-in commands."""
        builtin_commands = [
            HelpCommand(),
            ExitCommand(),
            ClearCommand(),
            HistoryCommand(),
            SessionsCommand(),
            ResumeCommand(),
            InfoCommand(),
            SearchCommand(),
        ]
        
        for cmd in builtin_commands:
            self._commands[cmd.name] = cmd
    
    def register_command(self, command: SlashCommand):
        """Register a custom command."""
        self._commands[command.name] = command
    
    def unregister_command(self, name: str):
        """Unregister a command."""
        if name in self._commands:
            del self._commands[name]
    
    def is_command(self, text: str) -> bool:
        """Check if text is a slash command."""
        return text.startswith('/') and len(text) > 1
    
    def parse_command(self, text: str) -> tuple[str, list[str]]:
        """Parse command text into command name and arguments."""
        if not self.is_command(text):
            raise ValueError("Not a slash command")
        
        parts = text[1:].split()  # Remove '/' and split
        command_name = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        return command_name, args
    
    async def execute_command(self, text: str, client: "StatefulCLIClient") -> CommandResult:
        """Execute a slash command."""
        try:
            command_name, args = self.parse_command(text)
            
            if command_name not in self._commands:
                return CommandResult(
                    command=command_name,
                    success=False,
                    result=None,
                    message=f"Unknown command: /{command_name}. Type /help for available commands."
                )
            
            command = self._commands[command_name]
            return await command.execute(args, client)
            
        except Exception as e:
            logger.error(f"Error executing command {text}: {e}")
            return CommandResult(
                command=text,
                success=False,
                result=None,
                message=f"Command failed: {e}"
            )