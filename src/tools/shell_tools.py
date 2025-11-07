# src/tools/shell_tools.py
"""
Module: src.tools.shell_tools

Provides the ShellTools class, a comprehensive utility wrapper for asynchronous
interaction with the local operating system, file system, and version control (Git).

This module serves as the primary interface for AI agents to gather project context
(e.g., file contents, directory structure, linter reports, Git history) and perform
safe, atomic file modifications within the project environment. It centralizes
configuration-driven access to external commands and file operations.
"""

# Standard Library Imports
import asyncio
import codecs
import logging
import os
import platform
import shlex
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

# Third-Party Library Imports
import httpx
from bs4 import BeautifulSoup

logger: logging.Logger = logging.getLogger(__name__)

# --- Constants for Default Configurations ---

DEFAULT_GIT_DIFF_COMMAND: list[str] = [
    "diff",
    "HEAD",
    "--patch-with-raw",
    "--minimal",
    "--patience",
    ":!uv.lock",
    ":!MANIFEST.in",
]
DEFAULT_GIT_LOG_COMMAND: list[str] = ["log", "-p"]
DEFAULT_ENCODING: str = "utf-8"
DEFAULT_COMMIT_NUMBER: int = 3
DEFAULT_TREE_DEPTH: int = 3
SUBPROCESS_TIMEOUT: int = 60


class ShellTools:
    """
    A class that encapsulates tools for shell and file system operations.

    This class provides an asynchronous interface for common tasks such as
    reading/writing files, running linters, fetching web content, and
    interacting with a Git repository. It is configured via a dictionary,
    typically loaded from a project's configuration file.
    """

    def __init__(
        self,
        agent: str,
        config: dict[str, Any],
        target_directory: Path | None = None,
    ) -> None:
        """
        Initializes the ShellTools instance.

        Args:
            agent: The name of the agent using the tools.
            config: A configuration dictionary for the tools.
            target_directory: The working directory for all operations.
                              If None, the current working directory is used.
        """
        self.configuration: dict[str, Any] = config
        self.agent_configuration: dict[str, Any] = self.configuration.get(agent, {})

        # --- Directory and File Matching Configuration ---
        self.design_docs: list[str] = self.configuration.get("design_docs", [])
        self.project_directories: list[str] = self.configuration.get(
            "project_directories", []
        )
        self.include_extensions: set[str] = set(
            self.configuration.get("include_extensions", [])
        )
        self.exclude_directories: set[str] = set(
            self.configuration.get("exclude_directories", [])
        )
        self.exclude_files: set[str] = set(self.configuration.get("exclude_files", []))

        # --- Command and Behavior Configuration ---
        self.encoding: str = self.configuration.get("encoding", DEFAULT_ENCODING)
        self.commit_number: int = self.configuration.get(
            "commit_number", DEFAULT_COMMIT_NUMBER
        )
        self.tree_depth: int = self.configuration.get("tree_depth", DEFAULT_TREE_DEPTH)
        self.git_diff_command: list[str] = self.configuration.get(
            "git_diff_command", DEFAULT_GIT_DIFF_COMMAND
        )
        self.git_log_command: list[str] = self.configuration.get(
            "git_log_command", DEFAULT_GIT_LOG_COMMAND
        )

        # --- Executable Path Verification ---
        git_executable_path: str | None = self._validate_executable("git")
        if not git_executable_path:
            logger.error(
                "Git command not found. Please ensure git is installed and in your PATH."
            )
            raise FileNotFoundError("Git executable not found.")
        self.git_executable: str = git_executable_path

        self.sync_executable: str | None = self._validate_executable("sync")
        if not self.sync_executable:
            logger.warning("'sync' command not found. File system sync may be skipped.")

        # --- Set Working Directory ---
        self.current_working_directory: Path = target_directory or Path.cwd()

    def _validate_executable(self, command: str) -> str | None:
        """
        Validates that the command is a known, safe executable and returns its path.
        This prevents B603 false positives by ensuring the executable is not user-controlled.
        """
        # Explicit allow-list of safe executables used in subprocess.run
        SAFE_EXECUTABLES = {"sync", "git", "tree"}
        if command not in SAFE_EXECUTABLES:
            logger.error("Attempted to use an unsafe executable: %s", command)
            return None

        executable_path = shutil.which(command)
        if not executable_path:
            return None

        # Ensure the path is absolute and points to a file
        path = Path(executable_path)
        if not path.is_absolute() or not path.is_file():
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
        environment = os.environ.copy()
        venv_bin_path = self.current_working_directory / ".venv" / "bin"
        if venv_bin_path.is_dir():
            environment["PATH"] = str(venv_bin_path) + os.pathsep + environment["PATH"]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.current_working_directory),
            env=environment,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError as error:
            command_str = shlex.join(command)
            logger.error("Command '%s' timed out.", command_str)
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
            raise RuntimeError("Command '%s' timed out." % command_str) from error

        stdout = stdout_bytes.decode(self.encoding, errors="ignore")
        stderr = stderr_bytes.decode(self.encoding, errors="ignore")

        if check and process.returncode != 0:
            command_str = shlex.join(command)
            error_message_template = (
                "Command '%s' failed with exit code %s:\nSTDOUT: %s\nSTDERR: %s"
            )
            logger.error(
                error_message_template,
                command_str,
                process.returncode,
                stdout,
                stderr,
            )
            # Format the message for the exception
            formatted_message = error_message_template % (
                command_str,
                process.returncode,
                stdout,
                stderr,
            )
            raise RuntimeError(formatted_message)
        return stdout, stderr, process.returncode

    def _is_in_excluded_directory(self, file_path: Path) -> bool:
        """Checks if a file path is within any of the configured excluded directories."""
        return any(part in self.exclude_directories for part in file_path.parts)

    def _should_exclude_file(self, file_name: str) -> bool:
        """
        Checks if a file name should be excluded based on configured patterns.

        Exclusion criteria include explicit file names defined in configuration,
        as well as common system/internal files (dotfiles starting with '.',
        or Python internal files starting with '__').
        """
        return (
            file_name in self.exclude_files
            or file_name.startswith(".")
            or file_name.startswith("__")
        )

    async def get_files_by_extensions(
        self, directory_path: Path, supported_extensions: list[str]
    ) -> list[Path]:
        """
        Retrieves a list of files within a directory that match the configured
        extensions and exclusion rules.

        This function wraps the synchronous file filtering logic in an asynchronous
        context, ensuring non-blocking I/O for the main application loop.

        Args:
            directory_path: The starting directory path to recursively search.
            supported_extensions: A list of file extensions to filter by.

        Returns:
            A list of Path objects for all matching files. Returns an empty list
            if the directory is invalid or an error occurs.
        """
        self.include_extensions = set(supported_extensions)
        if not directory_path.is_dir():
            logger.warning(
                "Directory does not exist or is not a directory: %s",
                directory_path,
            )
            return []

        try:
            matching_files = await asyncio.to_thread(
                list, self._get_filtered_files(directory_path)
            )
            logger.info(
                "Found %d matching files in %s",
                len(matching_files),
                directory_path,
            )
            return matching_files

        except OSError as error:
            logger.error(
                "Error searching directory %s: %s",
                directory_path,
                error,
                exc_info=True,
            )
            return []

    def _matches_extension(self, filename: str) -> bool:
        """Checks if a filename matches any of the configured included extensions."""
        if not self.include_extensions:
            return True
        return any(filename.endswith(ext) for ext in self.include_extensions)

    def _get_filtered_files(self, directory_path: Path) -> Iterator[Path]:
        """
        Yields files in a directory that match the configured criteria.
        This is a synchronous generator that performs file system I/O.
        """
        for item in directory_path.rglob("*"):
            if (
                item.is_file()
                and self._matches_extension(item.name)
                and not self._is_in_excluded_directory(item)
                and not self._should_exclude_file(item.name)
            ):
                yield item

    def _read_file_content(self, file_path: Path) -> str:
        """
        Synchronously reads and returns the content of a file, handling common
        file system errors and decoding issues.
        """
        try:
            with file_path.open(encoding=self.encoding, errors="ignore") as file_handle:
                return file_handle.read()
        except UnicodeDecodeError:
            logger.warning("Skipping binary file (UnicodeDecodeError): %s", file_path)
            return ""
        except FileNotFoundError:
            logger.error("File not found: %s", file_path)
            return ""
        except OSError as reading_error:
            logger.error("Error reading file %s: %s", file_path, reading_error)
            return ""

    async def read_file_content(self, file_path: Path) -> str:
        """
        Asynchronously reads and returns the content of a file.

        Args:
            file_path: The path to the file to read.

        Returns:
            The content of the file as a string, or an empty string on error.
        """
        return await asyncio.to_thread(self._read_file_content, file_path)

    def _write_file(
        self, filepath: Path, payload: str, backup: bool, atomic: bool
    ) -> bool:
        """
        Synchronously writes a payload to a file, handling directory creation,
        optional backups, and atomic operations.

        This internal method performs the actual blocking file I/O.

        Args:
            filepath: The relative path to the file to write.
            payload: The string content to write.
            backup: If True, creates a backup of the existing file before writing.
            atomic: If True, uses a temporary file and move operation for atomicity.

        Returns:
            True if the write was successful, False otherwise.
        """
        try:
            full_path = self.current_working_directory / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if backup and full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + ".bak")
                shutil.copy2(full_path, backup_path)
                logger.info("Created backup: %s", backup_path)

            if atomic:
                self._atomic_write(full_path, payload)
            else:
                full_path.write_text(payload, encoding=self.encoding)

            logger.info(
                "Successfully wrote %d characters to %s", len(payload), full_path.name
            )
            return True

        except (PermissionError, OSError) as file_error:
            logger.error(
                "File system error writing to %s: %s", filepath.name, file_error
            )
            return False
        except Exception as exception:
            logger.error(
                "Unexpected error writing to %s: %s",
                filepath.name,
                exception,
                exc_info=True,
            )
            return False

    async def write_file(
        self,
        filepath: Path,
        payload: str,
        backup: bool = False,
        atomic: bool = True,
    ) -> bool:
        """
        Asynchronously writes a payload to a file.

        Args:
            filepath: The path to the file to write.
            payload: The string content to write to the file.
            backup: If True, creates a backup of the existing file.
            atomic: If True, performs an atomic write operation.

        Returns:
            True if the write was successful, False otherwise.
        """
        return await asyncio.to_thread(
            self._write_file, filepath, payload, backup, atomic
        )

    def _atomic_write(self, target_path: Path, payload: str) -> None:
        """
        Performs an atomic write operation to a target path.

        This is achieved by writing to a temporary file in the same directory
        and then moving it to the target path.

        Args:
            target_path: The final destination path for the file.
            payload: The content to write.

        Raises:
            IOError: If the atomic write fails.
        """
        temp_file_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding=self.encoding,
                dir=target_path.parent,
                delete=False,
                prefix=f".tmp_{target_path.name}_",
                suffix=".tmp",
            ) as temp_file:
                temp_file_path = Path(temp_file.name)
                temp_file.write(payload)
                temp_file.flush()
                # On non-Windows systems, explicitly call 'sync' to ensure the temporary
                # file buffer is flushed to disk before the atomic move operation.
                # This increases robustness against system crashes during the write process.
                if platform.system() != "Windows" and self.sync_executable:
                    try:
                        subprocess.run(
                            [self.sync_executable], check=True, capture_output=True
                        )
                    except (
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                    ) as sync_error:
                        logger.warning(
                            "Failed to execute '%s' command: %s",
                            self.sync_executable,
                            sync_error,
                        )

            if target_path.exists():
                shutil.copymode(target_path, temp_file_path)

            shutil.move(temp_file_path, target_path)

        except OSError as error:
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise OSError(f"Atomic write to {target_path} failed") from error
        finally:
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)

    async def get_design_docs_content(self) -> str:
        """
        Asynchronously reads and concatenates the content of all configured design documents.

        It iterates through the paths defined in `self.design_docs`, reads each file
        concurrently, and joins the contents with double newlines.

        Returns:
            A single string containing the combined content of all found design documents.
        """
        tasks = []
        for doc_path_str in self.design_docs:
            full_path = self.current_working_directory / doc_path_str
            if full_path.is_file():
                tasks.append(self.read_file_content(full_path))
            else:
                logger.warning(
                    "Design document not found or is not a file: %s", full_path
                )

        contents = await asyncio.gather(*tasks)
        return "\n\n".join(filter(None, contents))

    @staticmethod
    async def fetch_urls_content(urls: list[str]) -> str:
        """
        Fetches content from a list of URLs concurrently and extracts clean text.

        Args:
            urls: A list of URL strings to fetch.

        Returns:
            A single string concatenating the clean text from all successfully fetched URLs.
        """
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            tasks = [client.get(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        content_parts: list[str] = []
        for url, result in zip(urls, results):
            if isinstance(result, httpx.Response) and result.status_code == 200:
                header = f"\n\n--- Content from: {url} ---\n\n"
                soup = BeautifulSoup(result.text, "html.parser")
                clean_text = soup.get_text()
                content_parts.append(header + clean_text)
                logger.info("Successfully fetched and parsed content from %s", url)
            else:
                error_msg = (
                    str(result)
                    if isinstance(result, Exception)
                    else f"Status code: {getattr(result, 'status_code', 'N/A')}"
                )
                logger.warning("Failed to fetch content from %s: %s", url, error_msg)

        return "\n".join(content_parts).strip()

    async def run_project_linters(self) -> str:
        """
        Runs all configured linters against the project concurrently and aggregates their output.

        This method handles cases where linters are missing or fail to execute.
        It uses `_run_command` with `check=False` because linters typically use
        non-zero exit codes to signal warnings/errors, not execution failure.

        Returns:
            A formatted string containing the combined output of all
            executed linters that reported issues, or a message indicating no issues were found.
        """
        linter_configs = self.configuration.get("linters", {})
        if not linter_configs:
            logger.info("No linters configured.")
            return "No linters configured."

        tasks = []
        for linter_name, linter_command_parts in linter_configs.items():
            command_str = shlex.join(linter_command_parts)
            logger.info("Running linter: %s with command: %s", linter_name, command_str)
            tasks.append(self._run_command(linter_command_parts, check=False))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        report_lines = []
        has_issues = False
        for linter_name, result in zip(linter_configs.keys(), results):
            report_section = []
            if isinstance(result, Exception):
                has_issues = True
                logger.error(
                    "Linter '%s' failed with an exception: %s", linter_name, result
                )
                report_section.append(
                    f"--- Linter: {linter_name} (Execution Error) ---"
                )
                report_section.append(
                    f"An exception occurred during execution: {result}"
                )
            else:
                # Result is a tuple: (stdout, stderr, return_code)
                if not isinstance(result, tuple):
                    raise TypeError(
                        f"Expected a tuple for linter result, but got {type(result).__name__}"
                    )
                standard_output, standard_error, exit_code = result

                output_parts = []
                if standard_output.strip():
                    output_parts.append(standard_output.strip())
                if standard_error.strip():
                    output_parts.append(standard_error.strip())
                combined_output = "\n".join(output_parts)

                if combined_output or exit_code != 0:
                    has_issues = True
                    report_section.append(
                        f"--- Linter: {linter_name} (Exit Code: {exit_code}) ---"
                    )
                    if combined_output:
                        report_section.append(combined_output)
                    else:
                        report_section.append(
                            "Linter finished with a non-zero exit code but produced no output."
                        )

            if report_section:
                report_lines.extend(report_section)
                report_lines.append("")  # Add a blank line for separation

        if not has_issues:
            return "No issues found by linters."

        return "\n".join(report_lines).strip()

    async def process_directory(self, directory_path: Path) -> str:
        """
        Recursively scans a directory, filters files based on configuration,
        reads their content, and concatenates them into a single string.

        Each file's content is prefixed with a header indicating its relative path,
        providing clear context for agents consuming this information.

        Args:
            directory_path: The root directory to start scanning.

        Returns:
            A single string containing the concatenated, contextualized content
            of all matching files.

        Raises:
            RuntimeError: If a critical OSError occurs during directory traversal.
        """
        if not directory_path.is_dir():
            logger.warning(
                "Directory does not exist or is not a directory: %s",
                directory_path,
            )
            return ""

        try:
            filtered_files = await asyncio.to_thread(
                list, self._get_filtered_files(directory_path)
            )

            tasks = [self.read_file_content(file_path) for file_path in filtered_files]
            file_contents = await asyncio.gather(*tasks)

            concatenated_content_parts: list[str] = []
            for file_path, file_content in zip(filtered_files, file_contents):
                if not file_content:
                    continue

                relative_path = file_path.relative_to(self.current_working_directory)
                header = f"\n\n--- File: {relative_path} ---\n\n"
                concatenated_content_parts.append(header + file_content)

            if not concatenated_content_parts:
                logger.info("No matching files found in directory: %s", directory_path)

            logger.info(
                "Concatenated %d files from %s",
                len(concatenated_content_parts),
                directory_path,
            )
            return "".join(concatenated_content_parts).strip()

        except OSError as error:
            logger.error(
                "Error processing directory %s: %s",
                directory_path,
                error,
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to process directory %s" % directory_path
            ) from error

    async def get_project_tree(self) -> str:
        """
        Generates a file and directory tree representation of the project structure.

        It first attempts to use the external 'tree' command for efficiency. If
        'tree' is unavailable or fails, it falls back to a recursive Python
        implementation (`_build_tree_lines`) executed in a separate thread to
        avoid blocking the event loop. The depth of the tree is controlled by
        `self.tree_depth`.

        Returns:
            A string representing the directory tree structure.
        """
        try:
            stdout, _, _ = await self._run_command(["tree", "-L", str(self.tree_depth)])
            return stdout.strip()
        except (FileNotFoundError, RuntimeError):
            logger.warning(
                "'tree' command not found or failed. Falling back to Python implementation."
            )

            def _build_tree_lines(dir_path: Path, level: int = 0) -> list[str]:
                if level >= self.tree_depth:
                    return []

                lines: list[str] = []
                try:
                    items = sorted(
                        list(dir_path.iterdir()),
                        key=lambda p: (p.is_file(), p.name.lower()),
                    )
                except OSError:
                    return []

                for i, item in enumerate(items):
                    if self._is_in_excluded_directory(
                        item
                    ) or self._should_exclude_file(item.name):
                        continue

                    is_last = i == len(items) - 1
                    prefix = "    " * level
                    connector = (
                        "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
                    )
                    lines.append(f"{prefix}{connector}{item.name}")

                    if item.is_dir():
                        lines.extend(_build_tree_lines(item, level + 1))
                return lines

            tree_lines = await asyncio.to_thread(
                _build_tree_lines, self.current_working_directory
            )
            return "\n".join(tree_lines)

    async def get_detected_languages(self) -> str:
        """
        Detects the primary programming languages used in the project by counting
        the occurrences of file extensions defined in the configuration.

        The file scanning is performed synchronously in a separate thread.

        Returns:
            A formatted string listing the top 5 most common file extensions
            and their counts.
        """

        def _scan_files() -> Counter[str]:
            extensions: list[str] = []
            for item in self._get_filtered_files(self.current_working_directory):
                if item.suffix:
                    extensions.append(item.suffix)
            return Counter(extensions)

        extension_counts = await asyncio.to_thread(_scan_files)

        if not extension_counts:
            return "No files with extensions found to analyze."

        most_common_extensions = extension_counts.most_common(5)
        return "\n".join(
            [
                f"{count} files with '{ext}' extension"
                for ext, count in most_common_extensions
            ]
        )

    async def get_git_info(self) -> dict[str, str | None]:
        """Asynchronously retrieves the git username and remote URL."""
        git_info: dict[str, str | None] = {"username": None, "url": None}
        try:
            username_cmd = [self.git_executable, "config", "user.name"]
            stdout, _, returncode = await self._run_command(username_cmd, check=False)
            if returncode == 0:
                git_info["username"] = stdout.strip()

            url_cmd = [self.git_executable, "config", "--get", "remote.origin.url"]
            stdout, _, returncode = await self._run_command(url_cmd, check=False)
            if returncode == 0:
                git_info["url"] = stdout.strip()

        except (RuntimeError, FileNotFoundError) as error:
            logger.error("Error getting git info: %s", error, exc_info=True)

        return git_info

    @staticmethod
    def cleanup_escapes(input_string: str) -> str:
        """
        Decodes unicode escape sequences in a string.

        Args:
            input_string: The string to decode.

        Returns:
            The decoded string, or the original string if decoding fails.
        """
        try:
            return codecs.decode(input_string, "unicode_escape")
        except UnicodeDecodeError:
            logger.warning(
                "Could not decode unicode escapes in string: %s...", input_string[:100]
            )
            return input_string

    async def get_git_context_for_patch(self) -> str:
        """
        Asynchronously executes configured Git commands to retrieve the current
        patch (diff) and recent commit history (log).

        The diff command is defined by `self.git_diff_command` and the log
        history depth is controlled by `self.commit_number`. Both commands run
        concurrently.

        Returns:
            A single string containing the concatenated output of the git diff
            followed by the git log.

        Raises:
            RuntimeError: If either of the underlying git commands fails to execute.
        """
        diff_cmd = [self.git_executable, *self.git_diff_command]
        log_cmd = [
            self.git_executable,
            *self.git_log_command,
            "-n",
            str(self.commit_number),
        ]

        logger.debug("Running git diff command: %s", shlex.join(diff_cmd))
        logger.debug("Running git log command: %s", shlex.join(log_cmd))

        try:
            diff_task = self._run_command(diff_cmd)
            log_task = self._run_command(log_cmd)
            diff_result, log_result = await asyncio.gather(diff_task, log_task)
            diff_stdout, _, _ = diff_result
            log_stdout, _, _ = log_result
            return f"{diff_stdout}\n{log_stdout}"
        except (RuntimeError, FileNotFoundError) as error:
            raise RuntimeError("Failed to create git patch context") from error

