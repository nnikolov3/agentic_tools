"""
Purpose:
Orchestrator for the Code Quality Enforcement workflow. This class manages file discovery,
runs the commentator agent, and implements a fault-tolerant loop that delegates
fixing of post-LLM validation errors to the developer agent.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

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
        """
        self.config: Dict[str, Any] = config
        self.mcp_name: str = mcp_name
        self.validation_service = ValidationService()
        self.shell_tools = ShellTools("code_quality_enforcer", config)

        # Configurable parameters
        self.max_fix_attempts: int = self.config.get("max_fix_attempts", 3)
        self.run_pre_validation: bool = self.config.get("run_pre_validation", True)
        self.file_extensions: List[str] = self.config.get("include_extensions", [".py"])

        # Base prompt for the developer agent to fix code
        self.developer_prompt: str = self.config.get("developer", {}).get("prompt", "")
        self.developer_agent_name: str = "developer"
        self.commentator_agent_name: str = "commentator"

    async def run(self, directory_path: str) -> None:
        """
        Orchestrates the code quality enforcement workflow for a directory.
        """
        logger.info(
            f"Starting code quality enforcement for directory: {directory_path}"
        )

        # 1. Discover Files
        source_files: List[Path] = self.shell_tools.get_files_by_extensions(
            directory_path, self.file_extensions
        )

        if not source_files:
            logger.warning(
                f"No source files found in {directory_path} matching extensions: {self.file_extensions}"
            )
            return

        logger.info(f"Found {len(source_files)} files to process.")

        # 2. Process Each File
        for file_path in source_files:
            await self._process_file(file_path)

    async def _process_file(self, file_path: Path) -> None:
        """
        Manages the validation, transformation, and fixing loop for a single file.
        """
        file_path_str = str(file_path)
        logger.info(f"\n--- Processing File: {file_path_str} ---")

        original_content: str = self.shell_tools.read_file_content_for_path(file_path)
        if not original_content:
            logger.error(f"Could not read content for {file_path_str}. Skipping.")
            return

        current_content: str = original_content

        # 3. Pre-LLM Validation (Optional)
        if self.run_pre_validation:
            pre_validation = self.validation_service.validate(current_content)
            if not pre_validation.is_valid:
                logger.warning(
                    f"File is 'dirty' before LLM transformation. Skipping {file_path_str}."
                )
                logger.debug(f"Pre-validation errors:\n{pre_validation.errors}")
                return

        # 4. Invoke 'commentator' Agent
        logger.info("Invoking 'commentator' agent for documentation update...")
        commentator_agent = self._create_agent(
            self.commentator_agent_name, current_content
        )
        transformed_content: str = await commentator_agent.run_agent()

        if not transformed_content:
            logger.error(
                f"Commentator returned empty content for {file_path_str}. Skipping save."
            )
            return

        current_content = transformed_content

        # 5. Post-LLM Validation & Fixer Loop
        for attempt in range(1, self.max_fix_attempts + 1):
            validation_result = self.validation_service.validate(current_content)

            if validation_result.is_valid:
                logger.info(f"Validation passed after {attempt} attempt(s).")

                # 6. Save Final File
                self.shell_tools.write_file(file_path_str, current_content)
                logger.info(f"Successfully saved clean file: {file_path_str}")
                return

            # Validation Failed - Need Fixer Agent
            if attempt >= self.max_fix_attempts:
                logger.error(
                    f"Validation failed after {self.max_fix_attempts} attempts. Discarding changes for {file_path_str}."
                )
                logger.debug(f"Final errors:\n{validation_result.errors}")
                return

            logger.warning(
                f"Validation failed (Attempt {attempt}/{self.max_fix_attempts}). Invoking 'developer' agent to fix..."
            )

            # Construct prompt for developer agent
            fixer_prompt = self._create_fixer_prompt(
                current_content, validation_result.errors
            )

            # Invoke 'developer' Agent
            developer_agent = self._create_agent(
                self.developer_agent_name, fixer_prompt
            )
            fixed_content: str = await developer_agent.run_agent()

            if not fixed_content:
                logger.error(
                    f"Developer agent returned empty content on attempt {attempt}. Terminating fix loop."
                )
                return

            current_content = fixed_content

        # Should not be reached, but as a safeguard
        logger.error(
            f"Fixer loop exited unexpectedly for {file_path_str}. Discarding changes."
        )

    def _create_agent(self, agent_name: str, chat_input: str) -> Agent:
        """Helper to create and initialize an Agent instance."""
        return Agent(
            configuration=self.config,
            agent_name=agent_name,
            project=self.mcp_name,
            chat=chat_input,
        )

    def _create_fixer_prompt(self, code: str, errors: str) -> str:
        """
        Creates the specialized prompt for the developer agent to fix validation errors.
        """
        # The developer agent's base prompt is already set up to expect this structure.
        return f"{self.developer_prompt}\n\n## FAILED CODE\n```python\n{code}\n```\n\n## VALIDATION ERRORS\n{errors}"
