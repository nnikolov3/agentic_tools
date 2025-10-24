"""
Purpose:
A stateless service responsible for running external validation tools (mypy, ruff, black)
on in-memory code content using temporary files. This isolates the complexity of
interacting with command-line tools and provides a structured result.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured result containing validation status and consolidated errors."""

    is_valid: bool
    errors: str  # Consolidated output from all failing tools


class ValidationService:
    """
    Runs mypy, ruff, and black on provided code content.
    """

    def __init__(self) -> None:
        """Initializes the validation service with the commands to be run."""
        # The commands are structured with a name and the command-line arguments.
        # `black --check` is used to see if formatting is needed without changing the file.
        self.commands: List[Tuple[str, List[str]]] = [
            ("black", ["black", "--check"]),
            ("ruff", ["ruff", "check"]),
            ("mypy", ["mypy", "--ignore-missing-imports"]),
        ]
    @staticmethod
    def _run_command(
        command_name: str,
        command_args: List[str],
        file_path: str,
    ) -> Tuple[bool, str]:
        """
        Executes a single validation command against the temporary file.

        Args:
            command_name: The friendly name of the tool (e.g., 'black').
            command_args: The list of command arguments (e.g., ['black', '--check']).
            file_path: The path to the temporary file to validate.

        Returns:
            A tuple containing a boolean for success and a string with any errors.
        """
        full_command = command_args + [file_path]
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=False,  # We handle the non-zero exit code manually.
                timeout=60,  # A generous timeout for static analysis.
            )

            # A non-zero return code indicates that the check failed.
            if result.returncode != 0:
                error_output = result.stdout + result.stderr
                return False, f"--- {command_name} Errors ---\n{error_output.strip()}"

            return True, ""

        except FileNotFoundError:
            error_msg = f"Error: Validation tool '{command_name}' not found. Please ensure it is installed and in your PATH."
            logger.error(error_msg)
            return False, error_msg
        except subprocess.TimeoutExpired:
            error_msg = (
                f"Error: Validation tool '{command_name}' timed out after 30 seconds."
            )
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = (
                f"An unexpected error occurred while running '{command_name}': {e}"
            )
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def validate(self, file_content: str) -> ValidationResult:
        """
        Runs all configured validation tools on the given code content.

        This method writes the content to a temporary file, runs each validation
        command against it, collects any errors, and cleans up the file.

        Args:
            file_content: A string containing the Python code to validate.

        Returns:
            A ValidationResult object indicating success or failure and containing
            consolidated error messages.
        """
        is_globally_valid = True
        all_errors: List[str] = []
        temp_file_path = ""

        try:
            # Create a temporary file to hold the content for the validation tools.
            # delete=False is necessary so we can pass its name to subprocesses.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Run each validation command sequentially.
            for command_name, command_args in self.commands:
                is_valid, error_message = self._run_command(
                    command_name, command_args, temp_file_path
                )
                if not is_valid:
                    is_globally_valid = False
                    all_errors.append(error_message)

        except Exception as e:
            logger.error(
                f"ValidationService failed during file operation: {e}", exc_info=True
            )
            return ValidationResult(
                is_valid=False, errors=f"Internal Validation Service Error: {e}"
            )
        finally:
            # Explicitly clean up the temporary file. This is critical.
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return ValidationResult(
            is_valid=is_globally_valid, errors="\n\n".join(all_errors)
        )
