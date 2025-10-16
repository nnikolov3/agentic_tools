"""
Unit tests for configurator functionality related to agent configuration validation
and prompt composition with skill tags.
"""
from __future__ import annotations

from pathlib import Path
import textwrap
import pytest

from src.configurator import Configurator


def _write_config(path: Path, content: str) -> None:
    """Helper to write configuration content to disk."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_get_agent_config_requires_skills(tmp_path: Path) -> None:
    """Ensure agent configurations must declare skills."""
    config_path = tmp_path / "mcp.toml"
    _write_config(
        config_path,
        """
        [multi-agent-mcp]
        project_name = "Test Project"
        project_description = "Sample description"

        [multi-agent-mcp.reviewer]
        prompt = "Review the changes"
        model_name = "test-model"
        model_providers = ["google"]
        temperature = 0.2
        description = "Test reviewer without skills"
        """,
    )

    configurator = Configurator(str(config_path))
    configurator.load()

    with pytest.raises(KeyError):
        configurator.get_agent_config("reviewer")


def test_combine_prompt_with_skills_injects_tags() -> None:
    """Skill tags should prefix prompts in a deterministic format."""
    base_prompt = "Provide feedback on the documentation quality."
    skills = ("documentation_review", "style_consistency", "accuracy_audit")

    # Create a real configurator instance (though we'll only use this method)
    configurator = Configurator.__new__(Configurator)  # Create without calling __init__
    combined_prompt = configurator.combine_prompt_with_skills(base_prompt, skills)

    assert combined_prompt.startswith("# Tags: documentation_review, style_consistency, accuracy_audit\n")
    assert combined_prompt.endswith(base_prompt)
