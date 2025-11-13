"""
Module: src.utils.shell_utils

Provides the ShellUtils class, a focused utility for executing pre-defined,
safe shell commands based on an agent's configuration. It serves as the
primary interface for agents to gather context by running external tools
like git, tree, etc., in a secure and asynchronous manner.
"""

# Standard Library Imports
import asyncio
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# --- Constants for Default Configurations ---
DEFAULT_ENCODING: str = "utf-8"
SUBPROCESS_TIMEOUT: int = 60


class ShellUtils:
    """
    A class that encapsulates utilities for shell and file system operations.

    This class provides a simplified, asynchronous interface for running
    shell commands specified in an agent's configuration. It ensures that
    only allow-listed executables can be run, enhancing security.
    """

    def __init__(
        self,
        agent_config: dict[str, Any],
        target_directory: Path | None = None,
    ) -> None:
        """
        Initializes the ShellUtils instance.

        Args:
            agent_config: A configuration dictionary for the specific agent.
            target_directory: The working directory for all operations.
                              If None, the current working directory is used.
        """
        self.agent_configuration: dict[str, Any] = agent_config
        self.encoding: str = self.agent_configuration.get("encoding", DEFAULT_ENCODING)
        self.current_working_directory: Path = target_directory or Path.cwd()

    def _validate_executable(self, command: str) -> str | None:
        """
        Validates that the command is a known, safe executable and returns its path.
        This prevents security risks by ensuring the executable is not user-controlled.
        """
        # Explicit allow-list of safe executables used in subprocess.run.
        # This is a critical security boundary.
        safe_executables = {"git", "tree", "ls", "cat", "grep"}
        if command not in safe_executables:
            logger.error("Attempted to use a non-allow-listed executable: %s", command)
            return None

        executable_path = shutil.which(command)
        if not executable_path:
            logger.warning("Executable not found in PATH: %s", command)
            return None

        path = Path(executable_path)
        if not path.is_absolute() or not path.is_file():
            logger.error("Invalid executable path for command '%s': %s", command, path)
            return None

        return executable_path

    async def _run_command(
        self,
        command: list[str],
        check: bool = True,
        timeout: int = SUBPROCESS_TIMEOUT,
    ) -> tuple[str, str, int | None]:
        """
        Asynchronously runs a shell command and returns its output.

        Args:
            command: The command to run as a list of strings.
            check: If True, raises an exception if the command returns a non-zero exit code.
            timeout: The timeout in seconds for the command.

        Returns:
            A tuple containing (stdout, stderr, return_code).

        Raises:
            RuntimeError: If the command fails and `check` is True, or if it times out.
        """
        validated_executable = self._validate_executable(command[0])
        if not validated_executable:
            raise ValueError(
                f"Command '{command[0]}' is not a valid or safe executable."
            )

        full_command = [validated_executable] + command[1:]
        command_str = shlex.join(full_command)
        logger.info("Executing command: %s", command_str)

        environment = os.environ.copy()
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.current_working_directory),
            env=environment,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError as error:
            logger.error("Command '%s' timed out.", command_str)
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
            raise RuntimeError(f"Command '{command_str}' timed out.") from error

        stdout = stdout_bytes.decode(self.encoding, errors="ignore")
        stderr = stderr_bytes.decode(self.encoding, errors="ignore")

        if check and process.returncode != 0:
            error_message = (
                f"Command '{command_str}' failed with exit code {process.returncode}:\n"
                f"STDOUT: {stdout}\nSTDERR: {stderr}"
            )
            logger.error(error_message)
            raise RuntimeError(error_message)

        return stdout, stderr, process.returncode

    async def execute_tool_commands(self) -> dict[str, Any]:
        """
        Executes all commands defined in the agent's 'tool_use' configuration.

        This method reads the `tool_use` dictionary from the agent's config,
        runs each command concurrently, and returns the results in a structured
        dictionary.

        Why: This approach makes the tool execution entirely data-driven. New
        tools can be added to an agent's capabilities by simply updating the
        configuration file, without any changes to the Python code.

        Returns:
            A dictionary where keys are the command names from the configuration
            (e.g., 'git_diff_command') and values are the stdout from the
            executed command.
        """
        tool_use_config = self.agent_configuration.get("tool_use", {})
        if not tool_use_config:
            logger.info(
                "No 'tool_use' commands configured for agent '%s'.",
                self.agent_configuration.get("name", "unknown"),
            )
            return {}

        tasks = {}
        for tool_name, command_args in tool_use_config.items():
            if isinstance(command_args, list):
                tasks[tool_name] = self._run_command(command_args, check=False)
            else:
                logger.warning(
                    "Skipping invalid tool command for '%s': not a list.", tool_name
                )

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        tool_outputs: dict[str, Any] = {}
        for tool_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(
                    "Command '%s' failed with an exception: %s", tool_name, result
                )
                tool_outputs[tool_name] = f"Error executing tool: {result}"
            elif isinstance(result, tuple):
                stdout, stderr, return_code = result
                # We prioritize stdout, but include stderr if stdout is empty
                output = stdout.strip() if stdout.strip() else stderr.strip()
                if return_code != 0:
                    logger.warning(
                        "Command '%s' exited with code %d. Output: %s",
                        tool_name,
                        return_code,
                        output,
                    )
                tool_outputs[tool_name] = output
            else:
                logger.error(
                    "Unexpected result type for tool '%s': %s", tool_name, type(result)
                )
                tool_outputs[tool_name] = (
                    "Error: Unexpected result type from command execution."
                )

        return tool_outputs
