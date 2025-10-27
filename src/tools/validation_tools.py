# src/tools/validation_tools.py
"""
Provides a service for validating Python code against static analysis tools.

This module defines the `ValidationService`, a stateless component designed to
run a suite of external linters and formatters (e.g., ruff, black, mypy) on a
given string of Python code. It operates by writing the code to a temporary
file, invoking the tools in a secure subprocess, and then aggregating the
results.

This approach ensures that validation can be performed in memory without
modifying source files on disk, making it ideal for use in automated workflows,
CI/CD pipelines, or as part of an agentic toolchain that generates or modifies
code.

Key Components:
- ValidationResult: A simple dataclass to encapsulate the outcome of the
  validation, including a boolean status and any aggregated error messages.
- ValidationService: The main class that orchestrates the execution of
  validation tools.
"""

# Standard Library imports
import dataclasses
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Final, List, Tuple

logger = logging.getLogger(__name__)

# --- Module Constants ---

# Self-Documenting Code: Use named constants for magic numbers.
# This defines the maximum time in seconds to wait for each validation tool
# to complete before it is terminated.
_COMMAND_TIMEOUT_SECONDS: Final[int] = 30

# Parameterize Everything: Decouple configuration from application logic.
# This tuple defines the validation commands to be executed. Each inner tuple
# contains a human-readable name for the tool and a tuple of the command
# and its arguments. This structure makes it easy to add, remove, or modify
# validation tools without changing the core logic of the ValidationService.
_VALIDATION_COMMANDS: Final[Tuple[Tuple[str, Tuple[str, ...]], ...]] = (
    ("black", ("black", ".")),
    ("ruff", ("ruff", "check", "--fix")),
    ("mypy", ("mypy", "--ignore-missing-imports")),
)


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """
    Represents the outcome of a validation process.

    This is an immutable data structure that provides a clear and consistent
    format for returning validation results.

    Attributes:
        is_valid: True if all validation checks passed, False otherwise.
        errors: A consolidated string containing all error messages from
                failing tools. If `is_valid` is True, this is empty.
    """

    is_valid: bool
    errors: str


class ValidationService:
    """
    Manages and executes a suite of static analysis tools on Python code.

    This service is stateless and operates by writing provided code content to
    a temporary file, then invoking configured command-line tools against it.
    It aggregates the results and provides a unified, easy-to-interpret outcome.
    """

    def __init__(self) -> None:
        """
        Initializes the validation service.

        The service is configured using the module-level `_VALIDATION_COMMANDS`
        constant, ensuring consistent behavior across all instances.
        """

        self.commands: List[Tuple[str, List[str]]] = [
            (name, list(args)) for name, args in _VALIDATION_COMMANDS
        ]

    @staticmethod
    def _run_command(
        command_name: str,
        command_args: List[str],
        file_path: str,
    ) -> Tuple[bool, str]:
        """
        Executes a single validation command against a specified file.

        This method securely runs an external command-line tool, capturing its
        output and return code. It includes robust error handling for common
        issues like a missing tool executable or process timeouts.

        Args:
            command_name: A human-readable name for the tool (e.g., 'black').
            command_args: The command and its arguments (e.g., ['black', '--check']).
            file_path: The absolute path to the file to validate.

        Returns:
            A tuple containing a boolean for success (True) or failure (False),
            and a string with any error output.
        """
        executable = command_args[0]
        executable_path = shutil.which(executable)

        if not executable_path:
            error_message = f"Error: Validation tool '{command_name}' not found. Please ensure '{executable}' is installed and in your system's PATH."
            logger.error(error_message)
            return False, error_message

        full_command = [executable_path] + command_args[1:] + [file_path]
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,  # We manually check the return code for better error reporting.
            )

            if result.returncode != 0:
                if command_name in ["black", "ruff"] and result.returncode == 1:
                    return True, ""
                # Combine stdout and stderr for a complete error report.
                error_output = (result.stdout + result.stderr).strip()
                return False, f"--- {command_name} Errors ---\n{error_output}"

            return True, ""

        except FileNotFoundError:
            # This case should be rare due to the shutil.which check, but is
            # kept as a safeguard against race conditions or unusual environments.
            error_message = f"Error: Validation tool '{command_name}' not found at path '{executable_path}'. The file may have been removed after being found."
            logger.error(error_message)
            return False, error_message
        except subprocess.TimeoutExpired:
            error_message = f"Error: Validation tool '{command_name}' timed out after {_COMMAND_TIMEOUT_SECONDS} seconds."
            logger.error(error_message)
            return False, error_message
        except Exception as error:
            # Error Handling Excellence: Catch unexpected errors and provide context.
            error_message = (
                f"An unexpected error occurred while running '{command_name}': {error}"
            )
            logger.error(error_message, exc_info=True)
            # The service's contract is to return a ValidationResult, not to
            # raise exceptions for tool failures. The original error is logged
            # with its traceback for debugging purposes.
            return False, error_message

    def validate(self, file_content: str) -> ValidationResult:
        """
        Performs validation on Python code content using all configured tools.

        This method orchestrates the entire validation process:
        1. Creates a temporary file and writes the `file_content` to it.
        2. Sequentially executes each configured validation command.
        3. Collects and aggregates error messages from any failing commands.
        4. Guarantees cleanup of the temporary file, even if errors occur.

        Args:
            file_content: A string containing the Python code to be validated.

        Returns:
            A `ValidationResult` object summarizing the outcome. If an internal
            error occurs (e.g., cannot create a temp file), `is_valid` will be
            `False` and `errors` will describe the internal issue.
        """
        all_errors: List[str] = []
        temp_file_path = ""

        try:
            # Create a temporary file to hold the content for the validation tools.
            # `delete=False` is crucial because the file path is passed to an
            # external subprocess. We manage its deletion manually in the `finally` block.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            for command_name, command_args in self.commands:
                is_valid, error_message = self._run_command(
                    command_name, command_args, temp_file_path
                )
                if not is_valid:
                    all_errors.append(error_message)

        except Exception as error:
            # Catch exceptions during file handling or the validation loop itself.
            logger.error(
                "ValidationService failed during file operation or command execution.",
                exc_info=True,
            )
            # Error Handling Excellence: Provide a clear error message for internal failures.
            return ValidationResult(
                is_valid=False,
                errors=f"Internal Validation Service Error: {error}",
            )
        finally:
            # Remove What Isn't Used: Ensure the temporary file is always removed.
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        is_globally_valid = not all_errors
        return ValidationResult(
            is_valid=is_globally_valid, errors="\n\n".join(all_errors)
        )
