# File: tests/test_prompt_utils.py
"""
Unit tests for the prompt_utils module.
"""

from src.prompt_utils import serialize_raw_response


def test_serialize_raw_response_none():
    """Test serialization of None value."""
    result = serialize_raw_response(None)
    assert result is None


def test_serialize_raw_response_with_model_dump():
    """Test serialization of object with model_dump method."""

    class MockObject:
        def model_dump(self, mode=None):
            return {"test": "value"}

    mock_obj = MockObject()
    result = serialize_raw_response(mock_obj)
    assert result == {"test": "value"}


def test_serialize_raw_response_with_dict_method():
    """Test serialization of object with dict method."""

    class MockObject:
        def dict(self):
            return {"test": "dict_value"}

    mock_obj = MockObject()
    result = serialize_raw_response(mock_obj)
    assert result == {"test": "dict_value"}


def test_serialize_raw_response_builtin_types():
    """Test serialization of built-in types."""
    # Test dict
    test_dict = {"key": "value"}
    assert serialize_raw_response(test_dict) == test_dict

    # Test list
    test_list = [1, 2, 3]
    assert serialize_raw_response(test_list) == test_list

    # Test string
    test_string = "hello"
    assert serialize_raw_response(test_string) == test_string

    # Test int
    test_int = 42
    assert serialize_raw_response(test_int) == test_int

    # Test float
    test_float = 3.14
    assert serialize_raw_response(test_float) == test_float

    # Test bool
    test_bool = True
    assert serialize_raw_response(test_bool) == test_bool


def test_serialize_raw_response_fallback_to_str():
    """Test serialization falls back to string conversion."""

    class CustomObject:
        def __str__(self):
            return "custom object string"

    obj = CustomObject()
    result = serialize_raw_response(obj)
    assert result == "custom object string"
