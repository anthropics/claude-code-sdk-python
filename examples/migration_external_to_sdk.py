#!/usr/bin/env python3
"""Migration Example: External MCP Server to SDK MCP Server.

This example demonstrates how to migrate from an external MCP server
(running as a separate process) to an in-process SDK MCP server.

Benefits of migration:
- No separate process management
- Better performance (no IPC overhead)
- Simpler deployment (single Python process)
- Easier debugging and testing
"""

import asyncio
from typing import Any, Dict

from claude_code_sdk import (
    ClaudeCodeOptions,
    create_sdk_mcp_server,
    query,
    tool,
)


# ============================================================================
# BEFORE: External MCP Server Approach
# ============================================================================

async def example_external_server():
    """Example of using an external MCP server (the old way)."""
    print("🔴 BEFORE: External MCP Server")
    print("=" * 60)
    
    # With external servers, you need to:
    # 1. Create a separate Python file for the MCP server
    # 2. Launch it as a subprocess
    # 3. Manage process lifecycle
    
    external_config = {
        "weather": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "weather_mcp_server"],  # Separate process
            "env": {"API_KEY": "your-api-key"}
        }
    }
    
    print("Configuration for external server:")
    print(f"  - Type: stdio (subprocess)")
    print(f"  - Command: python -m weather_mcp_server")
    print(f"  - Requires: Separate server file, process management")
    print(f"  - Overhead: IPC communication, process startup time")
    
    # This would require weather_mcp_server.py to be installed and available
    # options = ClaudeCodeOptions(mcp_servers=external_config)
    # async for msg in query(prompt="What's the weather?", options=options):
    #     ...
    
    print("\n⚠️  Issues with external servers:")
    print("  - Need to manage separate process lifecycle")
    print("  - IPC overhead for every tool call")
    print("  - Complex deployment (multiple files/processes)")
    print("  - Harder to debug across process boundaries")


# ============================================================================
# AFTER: SDK MCP Server Approach
# ============================================================================

# Define your tools directly in your application
@tool("get_weather", "Get current weather for a city", {"city": str})
async def get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather information for a city."""
    city = args["city"]
    
    # In a real implementation, you'd call a weather API here
    # For demo purposes, we'll return mock data
    weather_data = {
        "San Francisco": {"temp": 65, "condition": "Sunny"},
        "New York": {"temp": 72, "condition": "Cloudy"},
        "London": {"temp": 59, "condition": "Rainy"},
        "Tokyo": {"temp": 75, "condition": "Clear"},
    }
    
    weather = weather_data.get(city, {"temp": 70, "condition": "Unknown"})
    
    return {
        "content": [
            {
                "type": "text",
                "text": f"Weather in {city}: {weather['temp']}°F, {weather['condition']}"
            }
        ]
    }


@tool("get_forecast", "Get weather forecast", {"city": str, "days": int})
async def get_forecast(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather forecast for multiple days."""
    city = args["city"]
    days = min(args.get("days", 3), 7)  # Limit to 7 days
    
    # Mock forecast data
    forecast = []
    for day in range(1, days + 1):
        forecast.append(f"Day {day}: 70°F, Partly cloudy")
    
    return {
        "content": [
            {
                "type": "text",
                "text": f"Forecast for {city}:\n" + "\n".join(forecast)
            }
        ]
    }


async def example_sdk_server():
    """Example of using an SDK MCP server (the new way)."""
    print("\n\n🟢 AFTER: SDK MCP Server")
    print("=" * 60)
    
    # With SDK servers, everything is in-process
    weather_server = create_sdk_mcp_server(
        name="weather",
        version="1.0.0",
        tools=[get_weather, get_forecast]
    )
    
    print("Configuration for SDK server:")
    print(f"  - Type: sdk (in-process)")
    print(f"  - Command: None (runs in same process)")
    print(f"  - Requires: Just function definitions")
    print(f"  - Overhead: None (direct function calls)")
    
    # Use the SDK server
    options = ClaudeCodeOptions(
        mcp_servers={"weather": weather_server},
        allowed_tools=["get_weather", "get_forecast"],
        max_turns=1
    )
    
    print("\n✅ Benefits of SDK servers:")
    print("  - No process management needed")
    print("  - Zero IPC overhead")
    print("  - Simple deployment (single process)")
    print("  - Easy debugging (same process)")
    print("  - Better performance")
    
    # Actually query Claude with the SDK server
    print("\n📊 Testing SDK server with Claude:")
    print("-" * 40)
    
    prompt = "What's the weather in San Francisco and New York?"
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            for block in message.content:
                if hasattr(block, 'text'):
                    print(block.text)


# ============================================================================
# MIGRATION GUIDE
# ============================================================================

async def migration_guide():
    """Step-by-step migration guide."""
    print("\n\n📋 MIGRATION GUIDE")
    print("=" * 60)
    
    print("""
Step 1: Install the updated SDK
--------------------------------
pip install claude-code-sdk>=0.1.0  # Version with SDK MCP support

Step 2: Convert your external server to functions
-------------------------------------------------
# OLD: External server file (weather_server.py)
class WeatherServer(McpServer):
    def get_weather(self, city: str):
        return fetch_weather(city)

# NEW: Direct function in your app
@tool("get_weather", "Get weather", {"city": str})
async def get_weather(args):
    return fetch_weather(args["city"])

Step 3: Update your configuration
---------------------------------
# OLD: External server config
config = {
    "weather": {
        "type": "stdio",
        "command": "python",
        "args": ["weather_server.py"]
    }
}

# NEW: SDK server config
weather_server = create_sdk_mcp_server(
    name="weather",
    tools=[get_weather]
)
config = {"weather": weather_server}

Step 4: Remove process management code
--------------------------------------
# OLD: Need to manage server lifecycle
server_process = subprocess.Popen(...)
try:
    # ... use server ...
finally:
    server_process.terminate()

# NEW: No process management needed!
# Just use the server directly

Step 5: Test and deploy
-----------------------
- Run your tests (now simpler without subprocess mocking)
- Deploy as a single process (no separate server files)
- Monitor performance improvements
""")


async def main():
    """Run all examples."""
    # Show the before state
    await example_external_server()
    
    # Show the after state
    await example_sdk_server()
    
    # Show migration guide
    await migration_guide()
    
    print("\n" + "=" * 60)
    print("✅ Migration example completed!")
    print("\nFor more information, see the README and documentation.")


if __name__ == "__main__":
    asyncio.run(main())