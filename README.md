# Claude Code SDK for Python

Python SDK for Claude Code. See the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python) for more information.

## Installation

```bash
pip install claude-code-sdk
```

**Prerequisites:**
- Python 3.10+
- Node.js 
- Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Basic Usage: query()

`query()` is an async function for querying Claude Code. It returns an `AsyncIterator` of response messages. See [src/claude_code_sdk/query.py](src/claude_code_sdk/query.py).

```python
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock

# Simple query
async for message in query(prompt="Hello Claude"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)

# With options
options = ClaudeCodeOptions(
    system_prompt="You are a helpful assistant",
    max_turns=1
)

async for message in query(prompt="Tell me a joke", options=options):
    print(message)
```

### Using Tools

```python
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Write", "Bash"],
    permission_mode='acceptEdits'  # auto-accept file edits
)

async for message in query(
    prompt="Create a hello.py file", 
    options=options
):
    # Process tool use and results
    pass
```

### Working Directory

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## ClaudeSDKClient

`ClaudeSDKClient` supports bidirectional, interactive conversations with Claude
Code. See [src/claude_code_sdk/client.py](src/claude_code_sdk/client.py).

Unlike `query()`, `ClaudeSDKClient` additionally enables **custom tools** and **hooks**, both of which can be defined as Python functions.

### Custom Tools (as In-Process SDK MCP Servers)

A **custom tool** is a Python function that you can offer to Claude, for Claude to invoke as needed.

Custom tools are implemented in-process MCP servers that run directly within your Python application, eliminating the need for separate processes that regular MCP servers require.

For an end-to-end example, see [MCP Calculator](examples/mcp_calculator.py).

#### Creating a Simple Tool

```python
from claude_code_sdk import tool, create_sdk_mcp_server, ClaudeCodeOptions, ClaudeSDKClient

# Define a tool using the @tool decorator
@tool("greet", "Greet a user", {"name": str})
async def greet_user(args):
    return {
        "content": [
            {"type": "text", "text": f"Hello, {args['name']}!"}
        ]
    }

# Create an SDK MCP server
server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",
    tools=[greet_user]
)

# Use it with Claude
options = ClaudeCodeOptions(
    mcp_servers={"tools": server},
    allowed_tools=["mcp__tools__greet"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")

    # Extract and print response
    async for msg in client.receive_response():
        print(msg)
```

#### Benefits Over External MCP Servers

- **No subprocess management** - Runs in the same process as your application
- **Better performance** - No IPC overhead for tool calls
- **Simpler deployment** - Single Python process instead of multiple
- **Easier debugging** - All code runs in the same process
- **Type safety** - Direct Python function calls with type hints

#### Migration from External Servers

```python
# BEFORE: External MCP server (separate process)
options = ClaudeCodeOptions(
    mcp_servers={
        "calculator": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "calculator_server"]
        }
    }
)

# AFTER: SDK MCP server (in-process)
from my_tools import add, subtract  # Your tool functions

calculator = create_sdk_mcp_server(
    name="calculator",
    tools=[add, subtract]
)

options = ClaudeCodeOptions(
    mcp_servers={"calculator": calculator}
)
```

#### Mixed Server Support

You can use both SDK and external MCP servers together:

```python
options = ClaudeCodeOptions(
    mcp_servers={
        "internal": sdk_server,      # In-process SDK server
        "external": {                # External subprocess server
            "type": "stdio",
            "command": "external-server"
        }
    }
)
```

### Hooks

A **hook** is a Python function that the Claude Code *application* (*not* Claude) invokes at specific points of the Claude agent loop. Hooks can provide deterministic processing and automated feedback for Claude. Read more in [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks).

For more examples, see examples/hooks.py.

#### Example

```python
from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient, HookMatcher

async def check_bash_command(input_data, tool_use_id, context):
    tool_name = input_data["tool_name"]
    tool_input = input_data["tool_input"]
    if tool_name != "Bash":
        return {}
    command = tool_input.get("command", "")
    block_patterns = ["foo.sh"]
    for pattern in block_patterns:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Command contains invalid pattern: {pattern}",
                }
            }
    return {}

options = ClaudeCodeOptions(
    allowed_tools=["Bash"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command]),
        ],
    }
)

async with ClaudeSDKClient(options=options) as client:
    # Test 1: Command with forbidden pattern (will be blocked)
    await client.query("Run the bash command: ./foo.sh --help")
    async for msg in client.receive_response():
        print(msg)

    print("\n" + "=" * 50 + "\n")

    # Test 2: Safe command that should work
    await client.query("Run the bash command: echo 'Hello from hooks example!'")
    async for msg in client.receive_response():
        print(msg)
```


## Types

See [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py) for complete type definitions:
- `ClaudeCodeOptions` - Configuration options
- `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
- `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

## Error Handling

```python
from claude_code_sdk import (
    ClaudeSDKError,      # Base error
    CLINotFoundError,    # Claude Code not installed
    CLIConnectionError,  # Connection issues
    ProcessError,        # Process failed
    CLIJSONDecodeError,  # JSON parsing issues
)

try:
    async for message in query(prompt="Hello"):
        pass
except CLINotFoundError:
    print("Please install Claude Code")
except ProcessError as e:
    print(f"Process failed with exit code: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"Failed to parse response: {e}")
```

See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for all error types.

## Available Tools

See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list of available tools.

## Examples

See [examples/quick_start.py](examples/quick_start.py) for a complete working example.

See [examples/streaming_mode.py](examples/streaming_mode.py) for comprehensive examples involving `ClaudeSDKClient`. You can even run interactive examples in IPython from [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py).

## Development

### Prerequisites

- **Python 3.10+** - Required for modern type hints and async features
- **Node.js** - Required for Claude Code CLI
- **Git** - For version control and contributing
- **Claude Code CLI** - Install globally: `npm install -g @anthropic-ai/claude-code`

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anthropics/claude-code-sdk-python.git
   cd claude-code-sdk-python
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

   This installs the package in editable mode with all development dependencies including pytest, mypy, ruff, and coverage tools.

### Development Workflow

#### Code Quality Checks

Run these commands before committing code:

```bash
# Lint and fix issues automatically
python -m ruff check src/ tests/ --fix

# Format code
python -m ruff format src/ tests/

# Type checking (src/ only)
python -m mypy src/

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_client.py
```

#### Testing

**Unit Tests:**
```bash
# Run all unit tests with coverage
python -m pytest tests/ -v --cov=claude_code_sdk --cov-report=term-missing

# Run specific test class
python -m pytest tests/test_client.py::TestQueryFunction -v

# Run with different verbosity
python -m pytest tests/ -v  # verbose
python -m pytest tests/ -q  # quiet
```

**End-to-End Tests:**
```bash
# Requires ANTHROPIC_API_KEY environment variable
export ANTHROPIC_API_KEY="your-api-key"
python -m pytest e2e-tests/ -v -m e2e
```

**Example Scripts:**
```bash
# Test example scripts (requires API key)
python examples/quick_start.py
python examples/streaming_mode.py all
```

### Project Structure

```
claude-code-sdk-python/
├── src/claude_code_sdk/          # Main package
│   ├── __init__.py              # Public API exports
│   ├── client.py                # ClaudeSDKClient implementation
│   ├── query.py                 # One-shot query function
│   ├── types.py                 # Type definitions and data classes
│   ├── _errors.py               # Exception classes
│   └── _internal/               # Internal implementation
│       ├── client.py            # Internal client logic
│       ├── message_parser.py    # Message parsing utilities
│       ├── query.py             # Internal query implementation
│       └── transport/           # Transport layer
│           └── subprocess_cli.py # CLI subprocess management
├── tests/                       # Unit tests
├── e2e-tests/                   # End-to-end integration tests
├── examples/                    # Usage examples
├── pyproject.toml              # Project configuration
└── CLAUDE.md                   # Development instructions
```

### Code Standards

This project follows strict code quality standards:

#### Formatting and Linting (Ruff)
- **Line length:** 88 characters
- **Target Python:** 3.10+
- **Import sorting:** isort-compatible with first-party package recognition
- **Enabled rules:** pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear, flake8-comprehensions, flake8-use-pathlib, flake8-simplify

#### Type Checking (MyPy)
- **Strict mode:** Enabled for comprehensive type safety
- **Coverage:** All public functions must have type annotations
- **Configuration:** See `[tool.mypy]` in pyproject.toml

#### Testing Standards
- **Framework:** pytest with asyncio support
- **Coverage:** Minimum coverage enforced via CI
- **Test organization:** Separate files for different components
- **Async testing:** Uses anyio.run() for sync test compatibility

### Common Development Tasks

#### Running Examples Locally

```bash
# Basic query example
python examples/quick_start.py

# Interactive streaming with different backends
python examples/streaming_mode.py all

# Custom tools example
python examples/mcp_calculator.py

# IPython integration
python examples/streaming_mode_ipython.py
```

#### Working with Custom Tools

When developing custom MCP tools:

```python
from claude_code_sdk import tool, create_sdk_mcp_server

@tool("your_tool", "Description", {"param": str})
async def your_tool(args):
    # Your implementation
    return {"content": [{"type": "text", "text": "result"}]}
```

#### Debugging

```python
# Enable debug logging for transport layer
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with minimal example
from claude_code_sdk import query
async for msg in query("test", options=ClaudeCodeOptions(max_turns=1)):
    print(repr(msg))
```

### Troubleshooting

#### Common Issues

**"Claude Code CLI not found"**
```bash
# Verify installation
claude -v

# Reinstall if needed
npm install -g @anthropic-ai/claude-code
```

**"Module not found" errors**
```bash
# Ensure proper installation
pip install -e ".[dev]"

# Check Python path
python -c "import claude_code_sdk; print(claude_code_sdk.__file__)"
```

**Tests failing with async errors**
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check asyncio mode in pyproject.toml
grep -A 2 "\[tool.pytest-asyncio\]" pyproject.toml
```

**Type checking failures**
```bash
# Run mypy with verbose output
python -m mypy src/ --show-error-codes --show-error-context

# Check specific file
python -m mypy src/claude_code_sdk/client.py
```

#### API Key Issues

For e2e tests and examples requiring Claude API access:

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Or create a .env file (not tracked in git)
echo "ANTHROPIC_API_KEY=your-key" > .env
```

#### Performance Debugging

```python
# Monitor subprocess communication
import os
os.environ['CLAUDE_SDK_DEBUG'] = '1'

# Check for hanging processes
ps aux | grep claude
```

### Contributing Guidelines

1. **Before submitting a PR:**
   - Run the full test suite: `python -m pytest tests/`
   - Ensure code quality: `python -m ruff check src/ tests/ --fix`
   - Check types: `python -m mypy src/`
   - Test examples work: `python examples/quick_start.py`

2. **Commit message format:** Follow conventional commits
3. **Documentation:** Update docstrings for any API changes
4. **Tests:** Add tests for new functionality
5. **Examples:** Update examples if adding new features

## License

MIT
