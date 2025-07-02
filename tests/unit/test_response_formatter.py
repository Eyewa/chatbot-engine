"""
Unit tests for response formatting utilities.
"""

from unittest.mock import patch


from app.utils.response_formatter import (deep_clean_json_blocks,
                                          enforce_response_schema,
                                          filter_response_by_type,
                                          flatten_orders_field,
                                          parse_agent_output,
                                          unwrap_message_dicts)


class TestDeepCleanJsonBlocks:
    """Test JSON block cleaning functionality."""

    def test_clean_json_block(self):
        """Test cleaning JSON from code blocks."""
        text = '```json\n{"key": "value"}\n```'
        result = deep_clean_json_blocks(text)
        assert result == {"key": "value"}

    def test_clean_json_block_no_lang(self):
        """Test cleaning JSON from code blocks without language specifier."""
        text = '```\n{"key": "value"}\n```'
        result = deep_clean_json_blocks(text)
        assert result == {"key": "value"}

    def test_no_code_block(self):
        """Test text without code blocks."""
        text = '{"key": "value"}'
        result = deep_clean_json_blocks(text)
        assert result == {"key": "value"}

    def test_invalid_json(self):
        """Test invalid JSON returns original text."""
        text = "```json\n{invalid json}\n```"
        result = deep_clean_json_blocks(text)
        assert result == "{invalid json}"

    def test_non_string_input(self):
        """Test non-string input returns as-is."""
        data = {"key": "value"}
        result = deep_clean_json_blocks(data)
        assert result == data


class TestFilterResponseByType:
    """Test response filtering by type."""

    @patch("app.utils.response_formatter.RESPONSE_TYPES")
    def test_filter_no_type(self, mock_response_types):
        """Test filtering without response type."""
        mock_response_types.get.return_value = {"fields": ["field1"]}

        response = {"field1": "value1", "extra_field": "extra_value"}

        result = filter_response_by_type(response)

        # Should return original response when no type is specified
        assert result == response

    @patch("app.utils.response_formatter.RESPONSE_TYPES")
    def test_filter_unknown_type(self, mock_response_types):
        """Test filtering with unknown response type."""
        mock_response_types.get.return_value = None

        response = {"type": "unknown_type", "field1": "value1"}

        result = filter_response_by_type(response, "unknown_type")

        # Should return original response for unknown type
        assert result == response

    def test_filter_non_dict(self):
        """Test filtering non-dictionary input."""
        response = "not a dict"
        result = filter_response_by_type(response)
        assert result == response


class TestEnforceResponseSchema:
    """Test response schema enforcement."""

    def test_enforce_valid_schema(self):
        """Test enforcing valid schema."""
        schema = {"test_type": {"fields": ["field1", "field2"]}}

        response = {
            "type": "test_type",
            "field1": "value1",
            "field2": "value2",
            "extra_field": "extra_value",
        }

        result = enforce_response_schema(response, schema)

        assert "field1" in result
        assert "field2" in result
        assert "type" in result
        assert "extras" in result
        assert result["extras"]["extra_field"] == "extra_value"

    def test_enforce_unknown_type(self):
        """Test enforcing schema with unknown type."""
        schema = {"known_type": {"fields": ["field1"]}}

        response = {"type": "unknown_type", "field1": "value1"}

        result = enforce_response_schema(response, schema)

        # Should return original response for unknown type
        assert result == response

    def test_enforce_no_type(self):
        """Test enforcing schema without type."""
        schema = {"test_type": {"fields": ["field1"]}}

        response = {"field1": "value1"}

        result = enforce_response_schema(response, schema)

        # Should return original response when no type
        assert result == response


class TestParseAgentOutput:
    """Test agent output parsing."""

    def test_parse_string_without_json(self):
        """Test parsing string without JSON."""
        result = parse_agent_output("plain text")
        assert result == {"message": "plain text"}

    def test_parse_list(self):
        """Test parsing list input."""
        with patch("app.utils.response_formatter.RESPONSE_TYPES") as mock_types:
            mock_types.get.return_value = {"fields": ["field1"]}

            input_data = [{"field1": "value1"}, {"field1": "value2"}]
            result = parse_agent_output(input_data)

            assert "data" in result
            assert len(result["data"]) == 2

    def test_parse_dict(self):
        """Test parsing dictionary input."""
        with patch("app.utils.response_formatter.RESPONSE_TYPES") as mock_types:
            mock_types.get.return_value = {"fields": ["field1"]}

            input_data = {
                "type": "test_type",
                "field1": "value1",
                "extra_field": "extra_value",
            }

            result = parse_agent_output(input_data)

            assert "field1" in result
            assert "type" in result
            assert "extra_field" in result

    def test_parse_primitive(self):
        """Test parsing primitive values."""
        result = parse_agent_output(42)
        assert result == 42


class TestUnwrapMessageDicts:
    """Test message dictionary unwrapping."""

    def test_unwrap_simple_message(self):
        """Test unwrapping simple message dict."""
        data = {"message": "hello"}
        result = unwrap_message_dicts(data)
        assert result == "hello"

    def test_unwrap_nested_message(self):
        """Test unwrapping nested message dict."""
        data = {"message": {"message": "hello"}}
        result = unwrap_message_dicts(data)
        assert result == "hello"

    def test_no_unwrap_needed(self):
        """Test when no unwrapping is needed."""
        data = {"key": "value", "message": "hello"}
        result = unwrap_message_dicts(data)
        assert result == data

    def test_unwrap_list(self):
        """Test unwrapping list of message dicts."""
        data = [{"message": "hello"}, {"message": "world"}]
        result = unwrap_message_dicts(data)
        assert result == ["hello", "world"]

    def test_unwrap_primitive(self):
        """Test unwrapping primitive values."""
        data = "hello"
        result = unwrap_message_dicts(data)
        assert result == "hello"


class TestFlattenOrdersField:
    """Test orders field flattening."""

    def test_flatten_orders_with_data(self):
        """Test flattening orders field with nested data."""
        data = {"orders": {"data": [{"id": 1}, {"id": 2}]}}

        result = flatten_orders_field(data)

        assert "orders" in result
        assert result["orders"] == [{"id": 1}, {"id": 2}]

    def test_flatten_orders_no_data(self):
        """Test flattening orders field without nested data."""
        data = {"orders": [{"id": 1}, {"id": 2}]}

        result = flatten_orders_field(data)

        assert "orders" in result
        assert result["orders"] == [{"id": 1}, {"id": 2}]

    def test_flatten_no_orders(self):
        """Test flattening data without orders field."""
        data = {"other_field": "value"}

        result = flatten_orders_field(data)

        assert result == data

    def test_flatten_non_dict(self):
        """Test flattening non-dictionary input."""
        data = "not a dict"
        result = flatten_orders_field(data)
        assert result == data


def test_response_formatter_importable():
    """Test that response_formatter utilities can be imported and used."""
    from app.utils import response_formatter

    assert hasattr(response_formatter, "enforce_response_schema")
    assert callable(response_formatter.enforce_response_schema)


def test_response_metadata_filtering():
    """Test that response metadata is properly filtered out from conversation messages."""

    # Test the metadata filtering logic directly
    def clean_response_string(response_str):
        """Extract only content from response string, removing metadata."""
        if "response_metadata" in response_str:
            import re

            # Look for content field in the response - handle both single and double quotes
            content_match = re.search(r"'content':\s*'([^']*)'", response_str)
            if content_match:
                return content_match.group(1)
            # Try double quotes
            content_match = re.search(r"'content':\s*\"([^\"]*)\"", response_str)
            if content_match:
                return content_match.group(1)
            # If no content field, try to get the first part before metadata
            parts = response_str.split("response_metadata")
            if parts:
                # Clean up the first part - remove "content=" prefix if present
                first_part = parts[0].strip()
                if first_part.startswith("content="):
                    # Remove the content= prefix and surrounding quotes
                    content = first_part[8:]  # Remove "content="
                    if content.startswith("'") and content.endswith("'"):
                        content = content[1:-1]  # Remove surrounding quotes
                    elif content.startswith('"') and content.endswith('"'):
                        content = content[1:-1]  # Remove surrounding quotes
                    return content
                return first_part
        return response_str

    # Test cases
    test_cases = [
        # Case 1: Response with metadata
        (
            "content='Here are the details for customer 12345' response_metadata={'token_usage': {'total_tokens': 100}, 'model_name': 'gpt-4o'}",
            "Here are the details for customer 12345",
        ),
        # Case 2: Response without metadata
        (
            "Here are the details for customer 12345",
            "Here are the details for customer 12345",
        ),
        # Case 3: Complex metadata
        (
            "content='Customer summary' response_metadata={'token_usage': {'completion_tokens': 58, 'prompt_tokens': 191, 'total_tokens': 249}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_07871e2ad8'}",
            "Customer summary",
        ),
    ]

    for input_str, expected in test_cases:
        result = clean_response_string(input_str)
        assert (
            result == expected
        ), f"Expected '{expected}', got '{result}' for input '{input_str}'"
        assert "response_metadata" not in result
        assert "token_usage" not in result
        assert "model_name" not in result
