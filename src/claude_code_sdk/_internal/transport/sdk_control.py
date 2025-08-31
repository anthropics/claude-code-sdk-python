"""SDK MCP Transport Bridge.

This module implements a transport bridge that allows MCP servers running in the SDK process
to communicate with the Claude Code CLI process through control messages.

Architecture Overview:
    Unlike regular MCP servers that run as separate processes, SDK MCP servers run in-process
    within the SDK. This requires a special transport mechanism to bridge communication between:
    - The CLI process (where the MCP client runs)
    - The SDK process (where the SDK MCP server runs)

Message Flow:
    CLI → SDK:
    1. CLI's MCP Client calls a tool → sends JSONRPC request to SdkControlClientTransport
    2. Transport wraps the message in a control request with server_name and request_id
    3. Control request is sent via stdout to the SDK process
    4. SDK's StructuredIO receives the control response and routes it back to the transport
    5. Transport unwraps the response and returns it to the MCP Client
    
    SDK → CLI:
    1. Query receives control request with MCP message and calls transport.onmessage
    2. MCP server processes the message and calls transport.send() with response
    3. Transport calls sendMcpMessage callback with the response
    4. Query's callback resolves the pending promise with the response
    5. Query returns the response to complete the control request
"""

import asyncio
import json
from typing import Any, Callable, Dict, Optional

from mcp import JSONRPCMessage


class SdkControlClientTransport:
    """CLI-side transport for SDK MCP servers.
    
    This transport is used in the CLI process to bridge communication between:
    - The CLI's MCP Client (which wants to call tools on SDK MCP servers)
    - The SDK process (where the actual MCP server runs)
    
    It converts MCP protocol messages into control requests that can be sent
    through stdout/stdin to the SDK process.
    """
    
    def __init__(
        self,
        server_name: str,
        send_mcp_message: Callable[[str, JSONRPCMessage], asyncio.Future[JSONRPCMessage]],
    ):
        """Initialize the client transport.
        
        Args:
            server_name: Name of the SDK MCP server
            send_mcp_message: Callback to send messages and get responses
        """
        self.server_name = server_name
        self.send_mcp_message = send_mcp_message
        self._is_closed = False
        self.onclose: Optional[Callable[[], None]] = None
        self.onerror: Optional[Callable[[Exception], None]] = None
        self.onmessage: Optional[Callable[[JSONRPCMessage], None]] = None
    
    async def start(self) -> None:
        """Start the transport."""
        pass  # No initialization needed
    
    async def send(self, message: JSONRPCMessage) -> None:
        """Send a message to the SDK MCP server.
        
        Args:
            message: The JSONRPC message to send
            
        Raises:
            RuntimeError: If transport is closed
        """
        if self._is_closed:
            raise RuntimeError("Transport is closed")
        
        # Send the message and wait for the response
        response = await self.send_mcp_message(self.server_name, message)
        
        # Pass the response back to the MCP client
        if self.onmessage:
            self.onmessage(response)
    
    async def close(self) -> None:
        """Close the transport."""
        if self._is_closed:
            return
        self._is_closed = True
        if self.onclose:
            self.onclose()


class SdkControlServerTransport:
    """SDK-side transport for SDK MCP servers.
    
    This transport is used in the SDK process to bridge communication between:
    - Control requests coming from the CLI (via stdin)
    - The actual MCP server running in the SDK process
    
    It acts as a simple pass-through that forwards messages to the MCP server
    and sends responses back via a callback.
    
    Note: Query handles all request/response correlation and async flow.
    """
    
    def __init__(self, send_mcp_message: Callable[[JSONRPCMessage], None]):
        """Initialize the server transport.
        
        Args:
            send_mcp_message: Callback to send responses back
        """
        self.send_mcp_message = send_mcp_message
        self._is_closed = False
        self.onclose: Optional[Callable[[], None]] = None
        self.onerror: Optional[Callable[[Exception], None]] = None
        self.onmessage: Optional[Callable[[JSONRPCMessage], None]] = None
    
    async def start(self) -> None:
        """Start the transport."""
        pass  # No initialization needed
    
    async def send(self, message: JSONRPCMessage) -> None:
        """Send a response message back to the CLI.
        
        Args:
            message: The JSONRPC response to send
            
        Raises:
            RuntimeError: If transport is closed
        """
        if self._is_closed:
            raise RuntimeError("Transport is closed")
        
        # Simply pass the response back through the callback
        self.send_mcp_message(message)
    
    async def close(self) -> None:
        """Close the transport."""
        if self._is_closed:
            return
        self._is_closed = True
        if self.onclose:
            self.onclose()