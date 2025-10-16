# File: src/prompt_utils.py
"""
Shared utilities for common functionality across agents.
"""

from typing import Any


def serialize_raw_response(raw_response: Any) -> Any:
    """
    Serialize raw response for API return.
    
    Args:
        raw_response: Raw response from LLM API call
        
    Returns:
        Serialized response in appropriate format
    """
    if raw_response is None:
        return None
    if hasattr(raw_response, "model_dump"):
        try:
            return raw_response.model_dump(mode="json")
        except TypeError:
            # Handle different Pydantic versions that may not support the mode parameter
            return raw_response.model_dump()
    if hasattr(raw_response, "dict"):
        return raw_response.dict()
    if isinstance(raw_response, (dict, list, str, int, float, bool)):
        return raw_response
    return str(raw_response)