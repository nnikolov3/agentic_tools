"""
Purpose:
Orchestrator for the Code Quality Enforcement workflow. This class manages file discovery,
runs the commentator agent, and implements a fault-tolerant loop that delegates
fixing of post-LLM validation errors to the developer agent.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.agent import Agent
from src.tools.shell_tools import ShellTools
from src.tools.validation_tools import ValidationService

logger = logging.getLogger(__name__)


class CodeQualityEnforcer:
    """
    Manages the multi-step, multi-agent workflow for enforcing code quality standards.
    """

    def __init__(self, config: Dict[str, Any], mcp_name: str) -> None:
        """
        Initializes the orchestrator with configuration and services.

        Args:
            config: The full configuration dictionary for the project.
            mcp_name: The name of the main configuration project (e.g., 'agentic-tools').
        """
        self.config: Dict[str, Any] = config
        self.mcp_name: str = mcp_name
        self.validation_service = ValidationService()
        self.shell_tools = ShellTools("code_quality_enforcer", config)

        # Configurable parameters from the architect's plan.
        workflow_config = self.config.get("code_quality_enforcer", {})
        self.max_fix_attempts: int = workflow_config.get("max_fix_attempts", 3)
        self.run_pre_validation: bool = workflow_config.get("run_pre_validation", True)
        self.file_extensions: List[str] = self.config.get("include_extensions", [".py"])

        # Agent-specific details.
        self.developer_agent_name: str = "developer"
        self.commentator_agent_name: str = "commentator"
        self.developer_config: Dict[str, Any] = self.config.get(
            self.developer_agent_name, {}
        )
        self.developer_base_prompt: str = self.developer_config.get("prompt", "")

    async def run_on_directory(self, directory_path: str) -> None:
        """
        Orchestrates the code quality enforcement workflow for an entire directory.

        Args:
            directory_path: The path to the directory to process.
        """
        logger.info(
            f"Starting code quality enforcement for directory: {directory_path}"
        )

        source_files: List[Path] = self.shell_tools.get_files_by_extensions(
            directory_path, self.file_extensions
        )

        if not source_files:
            logger.warning(
                f"No source files found in '{directory_path}' matching extensions: {self.file_extensions}"
            )
            return

        logger.info(f"Found {len(source_files)} files to process.")

        for file_path in source_files:
            await self._process_file(file_path)

    async def run_on_file(self, file_path: str) -> None:
        """
        Orchestrates the code quality enforcement workflow for a single file.

        Args:
            file_path: The path to the file to process.
        """
        logger.info(f"Starting code quality enforcement for file: {file_path}")
        path_obj = Path(file_path)
        if not path_obj.exists() or not path_obj.is_file():
            logger.error(f"File not found or is not a regular file: {file_path}")
            return
        await self._process_file(path_obj)

    async def _run_validation_fix_loop(
        self, initial_content: str, context_description: str, file_path_str: str
    ) -> Tuple[bool, str]:
        """
        Runs a validation-and-fix loop for a given piece of code content.

        This method encapsulates the logic of repeatedly validating code and invoking
        a 'developer' agent to fix it until it passes or max attempts are reached.

        Args:
            initial_content: The starting code content to validate.
            context_description: A string describing the context (e.g., "pre-validation") for logging.
            file_path_str: The string representation of the file path for logging.

        Returns:
            A tuple containing:
            - bool: True if the content is valid by the end of the loop, False otherwise.
            - str: The final state of the code content.
        """
        current_content = initial_content
        for attempt in range(1, self.max_fix_attempts + 1):
            validation_result = self.validation_service.validate(current_content)
            if validation_result.is_valid:
                logger.info(
                    f"{context_description.capitalize()} passed for '{file_path_str}' after {attempt - 1} fix attempt(s)."
                )
                return True, current_content

            logger.warning(
                f"{context_description.capitalize()} failed for '{file_path_str}' (Attempt {attempt}/{self.max_fix_attempts})."
            )
            logger.debug(f"Validation errors:\n{validation_result.errors}")

            if attempt >= self.max_fix_attempts:
                break  # Exit loop; failure will be handled by the caller

            logger.info(
                f"Invoking '{self.developer_agent_name}' agent to fix {context_description} errors..."
            )
            fixer_input = self._create_fixer_prompt(
                current_content, validation_result.errors
            )
            developer_agent = self._create_agent(self.developer_agent_name, fixer_input)
            fixed_content: str | None = await developer_agent.run_agent()

            if not fixed_content or not isinstance(fixed_content, str):
                logger.error(
                    f"'{self.developer_agent_name}' returned empty content on {context_description} attempt {attempt}. Aborting."
                )
                return False, current_content  # Abort on agent failure

            current_content = self._strip_markdown_code_fence(fixed_content)

        # If the loop completes without returning, it means max attempts were reached.
        return False, current_content

    async def _process_file(self, file_path: Path) -> None:
        """
        Manages the validation, transformation, and fixing loop for a single file.

        This method implements the core state machine:
        Read -> Pre-Fix Loop -> Commentate -> Post-Fix Loop -> Save.
        """
        file_path_str = str(file_path)
        logger.info(f"\n--- Processing File: {file_path_str} ---")

        original_content: str = self.shell_tools.read_file_content_for_path(file_path)
        if not original_content:
            logger.error(f"Could not read content for {file_path_str}. Skipping.")
            return

        current_content: str = original_content

        # === STAGE 1: Pre-Validation (Mandatory Fix) ===
        if self.run_pre_validation:
            is_valid, current_content = await self._run_validation_fix_loop(
                initial_content=current_content,
                context_description="pre-validation",
                file_path_str=file_path_str,
            )
            if not is_valid:
                logger.error(
                    f"Maximum fix attempts reached during pre-validation for '{file_path_str}'. Skipping file."
                )
                return

            # Save the pre-fixed content to disk if it changed.
            if current_content != original_content:
                self.shell_tools.write_file(file_path_str, current_content)
                logger.info(f"Pre-fixed content saved to disk: {file_path_str}")
                # Update original_content for final comparison
                original_content = current_content

        # === STAGE 2: Commentator Agent ===
        logger.info(f"Invoking '{self.commentator_agent_name}' agent...")
        commentator_agent = self._create_agent(
            self.commentator_agent_name, current_content
        )
        commented_content: str | None = await commentator_agent.run_agent()

        if not commented_content or not isinstance(commented_content, str):
            logger.error(
                f"'{self.commentator_agent_name}' returned no content for '{file_path_str}'. Discarding changes."
            )
            return

        current_content = self._strip_markdown_code_fence(commented_content)

        # === STAGE 3: Post-Validation and Saving ===
        is_valid, final_content = await self._run_validation_fix_loop(
            initial_content=current_content,
            context_description="post-commentator validation",
            file_path_str=file_path_str,
        )

        if not is_valid:
            logger.error(
                f"Maximum fix attempts reached for '{file_path_str}'. Discarding all changes."
            )
            return

        # === STAGE 4: Final Save ===
        # Only save if the content changed from the content *before* the commentator ran.
        if final_content != original_content:
            self.shell_tools.write_file(file_path_str, final_content)
            logger.info(f"Successfully saved final updated file: {file_path_str}")
        else:
            logger.info(f"No further changes needed for {file_path_str}. File is clean.")

    def _create_agent(self, agent_name: str, chat_input: str) -> Agent:
        """Factory helper to create and initialize an Agent instance."""
        return Agent(
            configuration={self.mcp_name: self.config},
            agent_name=agent_name,
            project=self.mcp_name,
            chat=chat_input,
        )

    def _create_fixer_prompt(self, code: str, errors: str) -> str:
        """Creates the specialized prompt for the developer agent to fix validation errors."""
        return f"{self.developer_base_prompt}\n\n### FAILED CODE\n```python\n{code}\n```\n\n### VALIDATION ERRORS\n```\n{errors}\n```"

    def _strip_markdown_code_fence(self, content: str) -> str:
        """Removes the wrapping markdown code fence if present."""
        stripped_content = content.strip()
        if stripped_content.startswith("```") and stripped_content.endswith("```"):
            lines = stripped_content.split("\n")
            # Handle cases where the language is specified (e.g., ```python)
            if len(lines) > 1:
                return "\n".join(lines[1:-1])
        return content
