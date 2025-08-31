"""Claude SDK for Python."""

from dataclasses import dataclass
from typing import Any, Callable, Awaitable, TypeVar, Generic, Union

from ._errors import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from ._internal.transport import Transport
from .client import ClaudeSDKClient
from .query import query
from .types import (
    AssistantMessage,
    ClaudeCodeOptions,
    ContentBlock,
    McpServerConfig,
    McpSdkServerConfig,
    Message,
    PermissionMode,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# MCP Server Support

T = TypeVar('T')

@dataclass
class SdkMcpTool(Generic[T]):
    """Definition for an SDK MCP tool."""
    name: str
    description: str
    input_schema: Union[type[T], dict[str, Any]]
    handler: Callable[[T], Awaitable[dict[str, Any]]]


def tool(
    name: str,
    description: str,
    input_schema: Union[type, dict[str, Any]]
) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool]:
    """Decorator for defining MCP tools with type safety.
    
    Example:
        @tool("echo", "Echo input", {"input": str})
        async def echo_tool(args):
            return {"output": args["input"]}
    
    Args:
        name: Tool name
        description: Tool description  
        input_schema: Input schema as a type or dict
        
    Returns:
        Decorator function that creates an SdkMcpTool
    """
    def decorator(handler: Callable[[Any], Awaitable[dict[str, Any]]]) -> SdkMcpTool:
        return SdkMcpTool(name=name, description=description, input_schema=input_schema, handler=handler)
    return decorator


def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool] | None = None
) -> McpSdkServerConfig:
    """Create an in-process MCP server.
    
    Args:
        name: Server name
        version: Server version
        tools: List of tools to register
    
    Returns:
        MCP server configuration for use with Claude SDK
    
    Example:
        server = create_sdk_mcp_server(
            name="calculator",
            version="1.0.0",
            tools=[add_numbers, subtract_numbers]
        )
    """
    from mcp.server import Server
    
    # Create MCP server instance
    server = Server(name)
    
    # Register tools if provided
    if tools:
        for tool_def in tools:
            # Convert input_schema to appropriate format
            if isinstance(tool_def.input_schema, dict):
                schema = tool_def.input_schema
            else:
                # For TypedDict or other types, we need to convert to JSON schema
                # This is a simplified version - real implementation may need more sophisticated handling
                schema = {"type": "object", "properties": {}}
            
            # Register the tool with the server
            server.add_tool(
                name=tool_def.name,
                description=tool_def.description,
                input_schema=schema,
                handler=tool_def.handler
            )
    
    # Return SDK server configuration
    return McpSdkServerConfig(
        type="sdk",
        name=name,
        instance=server
    )

__version__ = "0.0.20"

__all__ = [
    # Main exports
    "query",
    # Transport
    "Transport",
    "ClaudeSDKClient",
    # Types
    "PermissionMode",
    "McpServerConfig",
    "McpSdkServerConfig",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    "Message",
    "ClaudeCodeOptions",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # MCP Server Support
    "create_sdk_mcp_server",
    "tool",
    "SdkMcpTool",
    # Errors
    "ClaudeSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "CLIJSONDecodeError",
]
