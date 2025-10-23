"""
Purpose:
A stateless service responsible for running external validation tools (mypy, ruff, black)
on in-memory code content using temporary files. This isolates the complexity of
interacting with command-line tools and provides a structured result.
"""

import subprocess
import tempfile
import os
import logging
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
        # Define the commands to run. Black is run first to format, then ruff and mypy check.
        self.commands: List[Tuple[str, List[str]]] = [
            ("black", ["black", "--check", "--diff"]),
            ("ruff", ["ruff", "check"]),
            ("mypy", ["mypy", "--ignore-missing-imports"]),
        ]

    def _run_command(
        self, command: str, args: List[str], file_path: str
    ) -> Tuple[bool, str]:
        """
        Executes a single validation command on the temporary file.
        """
        full_command = args + [file_path]
        try:
            # Use run_shell_command logic internally to execute the command
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=False,  # Do not raise exception on non-zero exit code
                timeout=10,
            )

            # Black and ruff return non-zero exit codes on failure. Mypy also returns non-zero.
            if result.returncode != 0:
                # For black, we only care if --check fails, which means it needs reformatting.
                # For ruff and mypy, non-zero means errors.
                error_output = result.stdout + result.stderr
                return False, f"--- {command} Errors ---\n{error_output}"

            return True, ""

        except FileNotFoundError:
            return (
                False,
                f"Error: Validation tool '{command}' not found. Please ensure it is installed and in your PATH.",
            )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"Error: Validation tool '{command}' timed out after 10 seconds.",
            )
        except Exception as e:
            return False, f"Error running validation tool '{command}': {e}"

    def validate(self, file_content: str) -> ValidationResult:
        """
        Runs mypy, ruff, and black on the given content and returns the result.
        """
        is_valid = True
        all_errors = []
        temp_file_path = ""

        try:
            # 1. Write content to a temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # 2. Run all validation commands
            for command_name, command_args in self.commands:
                success, error_msg = self._run_command(
                    command_name, command_args, temp_file_path
                )

                if not success:
                    is_valid = False
                    all_errors.append(error_msg)

        except Exception as e:
            logger.error(
                f"ValidationService failed during file operation: {e}", exc_info=True
            )
            return ValidationResult(
                is_valid=False, errors=f"Internal Validation Error: {e}"
            )
        finally:
            # 3. Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return ValidationResult(is_valid=is_valid, errors="\n".join(all_errors))
