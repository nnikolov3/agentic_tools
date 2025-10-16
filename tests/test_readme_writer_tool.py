"""
Test file for the ReadmeWriterTool module to ensure it follows design principles
and coding standards.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import os

from src.readme_writer_tool import ReadmeWriterTool
from src.configurator import Configurator


@pytest.fixture
def mock_configurator() -> MagicMock:
    """Fixture for a mocked Configurator."""
    from src.configurator import Configurator
    mock = MagicMock(spec=Configurator)
    mock.get_agent_config.return_value = {
        "prompt": "Test prompt for README generation",
        "model_name": "test_model",
        "model_providers": ["google"],
        "temperature": 0.5,
        "project_root": os.getcwd(),
        "skills": [
            "documentation_strategy",
            "technical_writing",
            "clarity_enforcement",
        ],
    }
    mock.get_context_policy.return_value = MagicMock(
        recent_minutes=10,
        include_extensions=(".py", ".md", ".toml"),
        exclude_dirs=(".git",),
        max_file_bytes=1024,
        max_total_bytes=4096,
        docs_paths=(),
        discovery=MagicMock(enabled=False, max_doc_bytes=1024),
    )
    # Configure the combine_prompt_with_skills method to call the real implementation
    real_configurator = Configurator.__new__(Configurator)
    mock.combine_prompt_with_skills = real_configurator.combine_prompt_with_skills
    return mock


def test_readme_writer_tool_instantiation(mock_configurator: MagicMock) -> None:
    """Test that the ReadmeWriterTool class can be instantiated."""
    readme_writer = ReadmeWriterTool(mock_configurator)
    assert isinstance(readme_writer, ReadmeWriterTool)


@patch("src.base_agent.api_caller")
@patch("src.shell_tools.collect_recent_sources")
@patch("pathlib.Path.iterdir")
@patch("pathlib.Path.exists")
def test_execute_success(
    mock_path_exists: MagicMock,
    mock_path_iterdir: MagicMock,
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method for a successful run."""
    # Mock the project root path structure
    mock_path_exists.return_value = True
    mock_collect_sources.return_value = "some source code"
    mock_api_caller.return_value = MagicMock(
        provider_name="google",
        model_name="test_model",
        content="# Project Title\n\nThis is a test README",
        raw_response={},
    )
    
    # Mock directory structure
    mock_dir = MagicMock()
    mock_dir.name = "src"
    mock_dir.is_dir.return_value = True
    mock_file = MagicMock()
    mock_file.name = "README.md"
    mock_file.is_dir.return_value = False
    mock_path_iterdir.return_value = [mock_dir, mock_file]

    readme_writer = ReadmeWriterTool(mock_configurator)
    result = readme_writer.execute(payload={})

    assert result["status"] == "success"
    assert result["data"]["provider"] == "google"
    assert "Project Title" in result["data"]["readme_content"]


@patch("src.base_agent.api_caller")
@patch("src.shell_tools.collect_recent_sources")
def test_skills_tags_in_system_prompt(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Ensure skill tags are injected into the system prompt."""
    captured_system_prompt: list[str] = []

    def fake_api_caller(config_payload: dict[str, object], messages: list[dict[str, str]]) -> MagicMock:
        captured_system_prompt.append(messages[0]["content"])
        return MagicMock(
            provider_name="google",
            model_name="test_model",
            content="# Project Title\n\nThis is a test README",
            raw_response={},
        )

    mock_collect_sources.return_value = "some source code"
    mock_api_caller.side_effect = fake_api_caller

    readme_writer = ReadmeWriterTool(mock_configurator)
    readme_writer.execute(payload={})

    assert captured_system_prompt, "System prompt was not captured"
    system_prompt = captured_system_prompt[0]
    assert system_prompt.startswith("# Tags:"), "System prompt should start with skill tags"

    expected_skills = mock_configurator.get_agent_config.return_value["skills"]
    for skill_name in expected_skills:
        assert skill_name in system_prompt, f"Missing skill tag '{skill_name}' in prompt"


@patch("src.base_agent.api_caller")
@patch("src.shell_tools.collect_recent_sources")
def test_execute_api_error(
    mock_collect_sources: MagicMock,
    mock_api_caller: MagicMock,
    mock_configurator: MagicMock,
) -> None:
    """Test the execute method when the API call fails."""
    mock_collect_sources.return_value = "some source code"
    mock_api_caller.return_value = None

    readme_writer = ReadmeWriterTool(mock_configurator)
    result = readme_writer.execute(payload={})

    assert result["status"] == "error"
    assert result["message"] == "No valid response received from any provider."


def test_readme_writer_tool_follows_design_principles(mock_configurator: MagicMock) -> None:
    """Test that the ReadmeWriterTool follows the design principles and coding standards."""
    # Instantiate the tool
    readme_writer = ReadmeWriterTool(mock_configurator)
    
    # Check that it has the required methods and properties
    assert hasattr(readme_writer, '_load_inputs')
    assert hasattr(readme_writer, '_assemble_docs')
    assert hasattr(readme_writer, '_assemble_sources')
    assert hasattr(readme_writer, '_assemble_project_structure')
    assert hasattr(readme_writer, '_assemble_config_info')
    assert hasattr(readme_writer, '_create_messages')
    assert hasattr(readme_writer, 'execute')
    
    # Check that the class follows single responsibility principle
    # It should have a clear, well-defined purpose as indicated by its name and docstring
    assert "intelligent readme writer" in readme_writer.__doc__.lower()
    assert "generates excellent documentation" in readme_writer.__doc__.lower()
    
    # Check that methods have descriptive names and follow explicit over implicit principle
    method_names = [method for method in dir(readme_writer) if callable(getattr(readme_writer, method)) and not method.startswith("__")]
    expected_methods = [
        'execute', 
        '_load_inputs', 
        '_assemble_docs', 
        '_assemble_sources', 
        '_assemble_project_structure', 
        '_assemble_config_info', 
        '_create_messages'
    ]
    
    for method in expected_methods:
        assert method in method_names
