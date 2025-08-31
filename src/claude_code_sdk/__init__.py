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
    
    Creates a tool that can be used with SDK MCP servers. The tool runs
    in-process within your Python application, providing better performance
    than external MCP servers.
    
    Args:
        name: Unique identifier for the tool. This is what Claude will use
            to reference the tool in function calls.
        description: Human-readable description of what the tool does.
            This helps Claude understand when to use the tool.
        input_schema: Schema defining the tool's input parameters.
            Can be either:
            - A dictionary mapping parameter names to types (e.g., {"text": str})
            - A TypedDict class for more complex schemas
            - A JSON Schema dictionary for full validation
    
    Returns:
        A decorator function that wraps the tool implementation and returns
        an SdkMcpTool instance ready for use with create_sdk_mcp_server().
    
    Example:
        Basic tool with simple schema:
        >>> @tool("greet", "Greet a user", {"name": str})
        ... async def greet(args):
        ...     return {"content": [{"type": "text", "text": f"Hello, {args['name']}!"}]}
        
        Tool with multiple parameters:
        >>> @tool("add", "Add two numbers", {"a": float, "b": float})
        ... async def add_numbers(args):
        ...     result = args["a"] + args["b"]
        ...     return {"content": [{"type": "text", "text": f"Result: {result}"}]}
        
        Tool with error handling:
        >>> @tool("divide", "Divide two numbers", {"a": float, "b": float})
        ... async def divide(args):
        ...     if args["b"] == 0:
        ...         return {"content": [{"type": "text", "text": "Error: Division by zero"}], "is_error": True}
        ...     return {"content": [{"type": "text", "text": f"Result: {args['a'] / args['b']}"}]}
    
    Notes:
        - The tool function must be async (defined with async def)
        - The function receives a single dict argument with the input parameters
        - The function should return a dict with a "content" key containing the response
        - Errors can be indicated by including "is_error": True in the response
    """
    def decorator(handler: Callable[[Any], Awaitable[dict[str, Any]]]) -> SdkMcpTool:
        return SdkMcpTool(name=name, description=description, input_schema=input_schema, handler=handler)
    return decorator


def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool] | None = None
) -> McpSdkServerConfig:
    """Create an in-process MCP server that runs within your Python application.
    
    Unlike external MCP servers that run as separate processes, SDK MCP servers
    run directly in your application's process. This provides:
    - Better performance (no IPC overhead)
    - Simpler deployment (single process)
    - Easier debugging (same process)
    - Direct access to your application's state
    
    Args:
        name: Unique identifier for the server. This name is used to reference
            the server in the mcp_servers configuration.
        version: Server version string. Defaults to "1.0.0". This is for
            informational purposes and doesn't affect functionality.
        tools: List of SdkMcpTool instances created with the @tool decorator.
            These are the functions that Claude can call through this server.
            If None or empty, the server will have no tools (rarely useful).
    
    Returns:
        McpSdkServerConfig: A configuration object that can be passed to
        ClaudeCodeOptions.mcp_servers. This config contains the server
        instance and metadata needed for the SDK to route tool calls.
    
    Example:
        Simple calculator server:
        >>> @tool("add", "Add numbers", {"a": float, "b": float})
        ... async def add(args):
        ...     return {"content": [{"type": "text", "text": f"Sum: {args['a'] + args['b']}"}]}
        >>> 
        >>> @tool("multiply", "Multiply numbers", {"a": float, "b": float})
        ... async def multiply(args):
        ...     return {"content": [{"type": "text", "text": f"Product: {args['a'] * args['b']}"}]}
        >>> 
        >>> calculator = create_sdk_mcp_server(
        ...     name="calculator",
        ...     version="2.0.0",
        ...     tools=[add, multiply]
        ... )
        >>> 
        >>> # Use with Claude
        >>> options = ClaudeCodeOptions(
        ...     mcp_servers={"calc": calculator},
        ...     allowed_tools=["add", "multiply"]
        ... )
        
        Server with application state access:
        >>> class DataStore:
        ...     def __init__(self):
        ...         self.items = []
        ... 
        >>> store = DataStore()
        >>> 
        >>> @tool("add_item", "Add item to store", {"item": str})
        ... async def add_item(args):
        ...     store.items.append(args["item"])
        ...     return {"content": [{"type": "text", "text": f"Added: {args['item']}"}]}
        >>> 
        >>> server = create_sdk_mcp_server("store", tools=[add_item])
    
    Notes:
        - The server runs in the same process as your Python application
        - Tools have direct access to your application's variables and state
        - No subprocess or IPC overhead for tool calls
        - Server lifecycle is managed automatically by the SDK
    
    See Also:
        - tool(): Decorator for creating tool functions
        - ClaudeCodeOptions: Configuration for using servers with query()
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
