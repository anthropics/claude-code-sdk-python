"""Tests for Claude SDK type definitions."""

from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookJSONOutput,
    ResultMessage,
)
from claude_agent_sdk.types import (
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


class TestHookTypes:
    """Test hook type definitions."""

    def test_hook_json_output_basic_usage(self):
        """Test basic usage: ensure a dict literal can be annotated as HookJSONOutput and used at runtime."""
        hook_output: HookJSONOutput = {"decision": "block"}
        assert hook_output["decision"] == "block"

    def test_hook_json_output_with_system_message(self):
        """Test HookJSONOutput with systemMessage field."""
        hook_output: HookJSONOutput = {
            "decision": "block",
            "systemMessage": "Not allowed",
        }
        assert hook_output["decision"] == "block"
        assert hook_output["systemMessage"] == "Not allowed"

    def test_hook_json_output_with_hook_specific(self):
        """Test HookJSONOutput with hookSpecificOutput field."""
        hook_output: HookJSONOutput = {"hookSpecificOutput": {"key": "value"}}
        assert hook_output["hookSpecificOutput"]["key"] == "value"

    def test_hook_json_output_all_fields(self):
        """Test HookJSONOutput with all possible fields."""
        hook_output: HookJSONOutput = {
            "decision": "block",
            "systemMessage": "Custom message",
            "hookSpecificOutput": {"key": "value"}
        }
        assert hook_output["decision"] == "block"
        assert hook_output["systemMessage"] == "Custom message"
        assert hook_output["hookSpecificOutput"]["key"] == "value"

    def test_hook_json_output_empty_dict(self):
        """Test HookJSONOutput with empty dict (all fields are optional)."""
        hook_output: HookJSONOutput = {}
        # Should not raise any errors - all fields are NotRequired
        assert isinstance(hook_output, dict)

    def test_hook_json_output_decision_only(self):
        """Test HookJSONOutput with only decision field."""
        hook_output: HookJSONOutput = {"decision": "block"}
        assert hook_output["decision"] == "block"
        assert "systemMessage" not in hook_output
        assert "hookSpecificOutput" not in hook_output

    def test_hook_json_output_system_message_only(self):
        """Test HookJSONOutput with only systemMessage field."""
        hook_output: HookJSONOutput = {"systemMessage": "Test message"}
        assert hook_output["systemMessage"] == "Test message"
        assert "decision" not in hook_output
        assert "hookSpecificOutput" not in hook_output

    def test_hook_json_output_hook_specific_only(self):
        """Test HookJSONOutput with only hookSpecificOutput field."""
        hook_output: HookJSONOutput = {"hookSpecificOutput": {"custom": "data"}}
        assert hook_output["hookSpecificOutput"]["custom"] == "data"
        assert "decision" not in hook_output
        assert "systemMessage" not in hook_output

    def test_hook_json_output_complex_hook_specific(self):
        """Test HookJSONOutput with complex hookSpecificOutput structure."""
        complex_data = {
            "hookEventName": "PreToolUse",
            "additionalContext": "Complex nested data",
            "nested": {
                "level1": {
                    "level2": ["item1", "item2", "item3"]
                }
            }
        }
        hook_output: HookJSONOutput = {"hookSpecificOutput": complex_data}
        assert hook_output["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        assert hook_output["hookSpecificOutput"]["nested"]["level1"]["level2"][0] == "item1"

    def test_hook_json_output_invalid_field_names(self):
        """Test that HookJSONOutput allows additional fields (TypedDict behavior)."""
        # TypedDict allows extra fields at runtime, but type checkers may warn
        hook_output: HookJSONOutput = {
            "decision": "block",
            "invalidField": "should be allowed at runtime",
            "anotherInvalid": 123
        }
        assert hook_output["decision"] == "block"
        assert hook_output["invalidField"] == "should be allowed at runtime"
        assert hook_output["anotherInvalid"] == 123

    def test_hook_json_output_type_constraints(self):
        """Test HookJSONOutput type constraints and runtime behavior."""
        # Note: TypedDict doesn't enforce runtime type checking, but documents expected types
        # These tests verify the structure can hold various types as documented
        
        # decision should be "block" or not present
        hook_output: HookJSONOutput = {"decision": "block"}
        assert hook_output["decision"] == "block"
        
        # systemMessage should be a string
        hook_output: HookJSONOutput = {"systemMessage": "Test message"}
        assert isinstance(hook_output["systemMessage"], str)
        
        # hookSpecificOutput can be any type (Any)
        hook_output: HookJSONOutput = {"hookSpecificOutput": "string"}
        assert hook_output["hookSpecificOutput"] == "string"
        
        hook_output: HookJSONOutput = {"hookSpecificOutput": 123}
        assert hook_output["hookSpecificOutput"] == 123
        
        hook_output: HookJSONOutput = {"hookSpecificOutput": None}
        assert hook_output["hookSpecificOutput"] is None

    def test_hook_json_output_integration_pretooluse_block(self):
        """Test HookJSONOutput integration pattern for PreToolUse blocking."""
        # Simulate a hook that blocks certain commands
        def mock_pretooluse_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> HookJSONOutput:
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})
            
            if tool_name == "Bash":
                command = tool_input.get("command", "")
                if "rm -rf" in command:
                    return {
                        "decision": "block",
                        "systemMessage": "Dangerous command blocked",
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "Command contains dangerous pattern: rm -rf"
                        }
                    }
            return {}
        
        # Test blocking case
        result = mock_pretooluse_hook(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
            "tool-123",
            None
        )
        assert result["decision"] == "block"
        assert result["systemMessage"] == "Dangerous command blocked"
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        
        # Test non-blocking case
        result = mock_pretooluse_hook(
            {"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
            "tool-123",
            None
        )
        assert result == {}

    def test_hook_json_output_integration_session_start(self):
        """Test HookJSONOutput integration pattern for SessionStart hook."""
        def mock_session_start_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> HookJSONOutput:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "Custom session instructions",
                    "userPreferences": {
                        "theme": "dark",
                        "language": "python"
                    }
                }
            }
        
        result = mock_session_start_hook({}, None, None)
        assert "decision" not in result
        assert "systemMessage" not in result
        assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert result["hookSpecificOutput"]["additionalContext"] == "Custom session instructions"
        assert result["hookSpecificOutput"]["userPreferences"]["theme"] == "dark"

    def test_hook_json_output_integration_user_prompt_submit(self):
        """Test HookJSONOutput integration pattern for UserPromptSubmit hook."""
        def mock_user_prompt_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> HookJSONOutput:
            prompt = input_data.get("prompt", "")
            
            if "password" in prompt.lower():
                return {
                    "systemMessage": "Warning: Prompt contains sensitive information",
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "securityWarning": "Prompt may contain sensitive data",
                        "recommendation": "Consider removing sensitive information"
                    }
                }
            return {}
        
        # Test warning case
        result = mock_user_prompt_hook(
            {"prompt": "What is my password for the system?"},
            None,
            None
        )
        assert result["systemMessage"] == "Warning: Prompt contains sensitive information"
        assert result["hookSpecificOutput"]["securityWarning"] == "Prompt may contain sensitive data"
        
        # Test normal case
        result = mock_user_prompt_hook(
            {"prompt": "How do I create a new file?"},
            None,
            None
        )
        assert result == {}


class TestMessageTypes:
    """Test message type creation and validation."""

    def test_user_message_creation(self):
        """Test creating a UserMessage."""
        msg = UserMessage(content="Hello, Claude!")
        assert msg.content == "Hello, Claude!"

    def test_assistant_message_with_text(self):
        """Test creating an AssistantMessage with text content."""
        text_block = TextBlock(text="Hello, human!")
        msg = AssistantMessage(content=[text_block], model="claude-opus-4-1-20250805")
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello, human!"

    def test_assistant_message_with_thinking(self):
        """Test creating an AssistantMessage with thinking content."""
        thinking_block = ThinkingBlock(thinking="I'm thinking...", signature="sig-123")
        msg = AssistantMessage(
            content=[thinking_block], model="claude-opus-4-1-20250805"
        )
        assert len(msg.content) == 1
        assert msg.content[0].thinking == "I'm thinking..."
        assert msg.content[0].signature == "sig-123"

    def test_tool_use_block(self):
        """Test creating a ToolUseBlock."""
        block = ToolUseBlock(
            id="tool-123", name="Read", input={"file_path": "/test.txt"}
        )
        assert block.id == "tool-123"
        assert block.name == "Read"
        assert block.input["file_path"] == "/test.txt"

    def test_tool_result_block(self):
        """Test creating a ToolResultBlock."""
        block = ToolResultBlock(
            tool_use_id="tool-123", content="File contents here", is_error=False
        )
        assert block.tool_use_id == "tool-123"
        assert block.content == "File contents here"
        assert block.is_error is False

    def test_result_message(self):
        """Test creating a ResultMessage."""
        msg = ResultMessage(
            subtype="success",
            duration_ms=1500,
            duration_api_ms=1200,
            is_error=False,
            num_turns=1,
            session_id="session-123",
            total_cost_usd=0.01,
        )
        assert msg.subtype == "success"
        assert msg.total_cost_usd == 0.01
        assert msg.session_id == "session-123"


class TestOptions:
    """Test Options configuration."""

    def test_default_options(self):
        """Test Options with default values."""
        options = ClaudeAgentOptions()
        assert options.allowed_tools == []
        assert options.system_prompt is None
        assert options.permission_mode is None
        assert options.continue_conversation is False
        assert options.disallowed_tools == []

    def test_claude_code_options_with_tools(self):
        """Test Options with built-in tools."""
        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Edit"], disallowed_tools=["Bash"]
        )
        assert options.allowed_tools == ["Read", "Write", "Edit"]
        assert options.disallowed_tools == ["Bash"]

    def test_claude_code_options_with_permission_mode(self):
        """Test Options with permission mode."""
        options = ClaudeAgentOptions(permission_mode="bypassPermissions")
        assert options.permission_mode == "bypassPermissions"

        options_plan = ClaudeAgentOptions(permission_mode="plan")
        assert options_plan.permission_mode == "plan"

        options_default = ClaudeAgentOptions(permission_mode="default")
        assert options_default.permission_mode == "default"

        options_accept = ClaudeAgentOptions(permission_mode="acceptEdits")
        assert options_accept.permission_mode == "acceptEdits"

    def test_claude_code_options_with_system_prompt_string(self):
        """Test Options with system prompt as string."""
        options = ClaudeAgentOptions(
            system_prompt="You are a helpful assistant.",
        )
        assert options.system_prompt == "You are a helpful assistant."

    def test_claude_code_options_with_system_prompt_preset(self):
        """Test Options with system prompt preset."""
        options = ClaudeAgentOptions(
            system_prompt={"type": "preset", "preset": "claude_code"},
        )
        assert options.system_prompt == {"type": "preset", "preset": "claude_code"}

    def test_claude_code_options_with_system_prompt_preset_and_append(self):
        """Test Options with system prompt preset and append."""
        options = ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": "Be concise.",
            },
        )
        assert options.system_prompt == {
            "type": "preset",
            "preset": "claude_code",
            "append": "Be concise.",
        }

    def test_claude_code_options_with_session_continuation(self):
        """Test Options with session continuation."""
        options = ClaudeAgentOptions(continue_conversation=True, resume="session-123")
        assert options.continue_conversation is True
        assert options.resume == "session-123"

    def test_claude_code_options_with_model_specification(self):
        """Test Options with model specification."""
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-5", permission_prompt_tool_name="CustomTool"
        )
        assert options.model == "claude-sonnet-4-5"
        assert options.permission_prompt_tool_name == "CustomTool"
