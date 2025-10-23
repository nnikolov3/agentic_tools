"""
Purpose:
This module implements shell-based tools for agent interactions, including file operations,
git integration, and directory processing. It follows foundational principles by handling
configurations explicitly, providing robust error logging, and ensuring atomic operations
where critical to prevent data corruption in file writes.
"""

import codecs
import logging
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List
from typing import Optional, Any

# Self-Documenting Code: Dedicated logger for shell operations traceability.
logger: logging.Logger = logging.getLogger(__name__)


class ShellTools:
    """
    Handles shell-related operations such as file reading/writing, git interactions, and
    directory traversal for project context gathering. Emphasizes atomic writes and
    configurable filtering for efficiency in large codebases.
    """

    def __init__(self, agent: str, config: Dict[str, Any]) -> None:
        """
        Initializes the ShellTools with agent-specific and global configuration.

        Extracts relevant settings for git, directories, extensions, and exclusions to
        enable focused operations without repeated config lookups.

        Args:
            agent: The agent name for sub-config selection.
            config: The full configuration dictionary from TOML.
        """
        # Explicit Over Implicit: Store config subsets directly for performance.
        self.config: Dict[str, Any] = config
        self.agent_config: Dict[str, Any] = self.config.get(agent, {})
        self.git_diff_command: List[str] = self.config.get(
            "git_diff_command",
            ["git", "diff", "--patch-with-raw", "--minimal", "--patience"],
        )
        self.design_docs: List[str] = config.get("design_docs", [])
        self.source: List[str] = config.get("source", [])
        self.project_directories: List[str] = config.get("project_directories", [])
        self.include_extensions: List[str] = config.get("include_extensions", [])
        self.max_file_bytes: Optional[int] = config.get("max_file_bytes")
        self.exclude_directories: List[str] = config.get("exclude_directories", [])
        self.exclude_files: List[str] = config.get("exclude_files", [])
        self.recent_minutes: Optional[int] = config.get("recent_minutes")

        # Internal state initialization for file operations.
        self.filepath: Path = Path(os.getcwd())
        self.filename: str = ""
        self.payload: str = ""
        self.encoding: str = "utf-8"
        self.backup: bool = False
        self.atomic: bool = True
        self.backup_path: Path = Path()
        self.root_path: Optional[Path] = None
        self.directory_tree: Dict[str, Any] = {}
        self.current_directory_level: Dict[str, Any] = {}
        self.project_root_path: Path = Path(os.getcwd())
        self.directories: List[Path] = []
        self.files: List[Path] = []
        self.relative_path: Optional[Path] = None
        self.current_working_directory: Path = self.project_root_path

    def concatenate_all_files(self) -> Dict[str, str]:
        """
        Traverse the project directories and concatenate all files.

        Processes project directories to build concatenated content, excluding specified
        patterns for focused context.

        Returns:
            Dictionary of concatenated file contents keyed by directory type.
        """
        concatenated_files_dict: Dict[str, str] = {}

        # Process all project directories if configured.
        if self.project_directories:
            concatenated_files_dict.update(
                self._process_directories(self.project_directories, "project")
            )

        return concatenated_files_dict

    def process_directory(self, directory_path: str) -> str:
        """
        Process a directory, concatenate the contents of all text files, and return as a single string.

        This function traverses the given directory, reads each text file, and combines
        their contents into a single string, with each file's content prefixed by a
        header indicating its relative path. This is useful for providing context to LLMs
        in a portable way.

        Args:
            directory_path: The absolute path to the directory to process.

        Returns:
            A string containing the concatenated contents of all text files,
            or an empty string if the directory does not exist or contains no matching files.
        """
        try:
            root_dir: Path = Path(directory_path)
            if not root_dir.is_dir():
                logger.warning(
                    f"Directory does not exist or is not a directory: {directory_path}"
                )
                return ""

            concatenated_content: str = ""
            file_count: int = 0

            for item in root_dir.rglob("*"):
                if (
                    item.is_file()
                    and self._matches_extension(item.name)
                    and item.name not in self.exclude_files
                ):
                    # Check if the file is in an excluded directory.
                    if any(
                        excluded in item.parts for excluded in self.exclude_directories
                    ):
                        continue

                    file_payload: str = self.read_file_content_for_path(item)
                    logger.info(f"Reading file: {item}")

                    # Skip files that returned an error message (e.g., "") or are binary.
                    if file_payload.strip().startswith("<"):
                        continue

                    relative_path: Path = item.relative_to(root_dir)
                    header: str = f"\n\n--- File: {relative_path} ---\n\n"
                    concatenated_content += header + file_payload
                    file_count += 1

            if file_count == 0:
                logger.info(f"No matching files found in directory: {directory_path}")

            logger.info(f"Concatenated {file_count} files from {directory_path}")
            return concatenated_content.strip()

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return ""

    def get_design_docs_content(self) -> str:
        """
        Reads and concatenates the content of all specified design documents.

        Returns:
            Concatenated string of design document contents.
        """
        design_docs_content: str = ""
        design_docs_paths: List[str] = self.config.get("design_docs", [])

        for doc_path in design_docs_paths:
            full_path: Path = Path(f"{self.current_working_directory}/{doc_path}")
            if full_path.exists() and full_path.is_file():
                design_docs_content += self.read_file_content_for_path(full_path)
            else:
                logger.warning(f"Design document not found: {full_path}")

        return design_docs_content

    def _process_directories(
        self, directories: List[str], directory_type: str
    ) -> Dict[str, str]:
        """
        Helper method to process a list of directories and build their file trees.

        Args:
            directories: List of directory paths to process.
            directory_type: Type label for logging ("source" or "project").

        Returns:
            Dictionary containing concatenated file contents.
        """
        result: Dict[str, str] = {}

        for directory in directories:
            logger.info(f"Processing {directory_type} directory: {directory}")
            self.project_root_path = Path(directory)

            if not self.project_root_path.exists():
                logger.warning(
                    f"{directory_type.capitalize()} directory does not exist: {directory}"
                )
                continue

            result.update(self._build_directory_tree())

        return result

    def _build_directory_tree(self) -> Dict[str, str]:
        """
        Build nested dictionary structure for a directory tree.

        Returns:
            Nested dictionary representing the directory tree with file contents.
        """
        self.directory_tree = {}

        for project_root, directories, files in os.walk(self.project_root_path):
            # Exclude directories in-place to prevent os.walk from descending into them.
            directories[:] = [
                directory
                for directory in directories
                if directory not in self.exclude_directories
            ]

            # Calculate relative path from root.
            self.relative_path = Path(project_root).relative_to(self.project_root_path)

            # Navigate to correct position in nested dict.
            self.current_directory_level = self.directory_tree
            if self.relative_path and str(self.relative_path) != ".":
                for part in self.relative_path.parts:
                    if part not in self.current_directory_level:
                        self.current_directory_level[part] = {}
                    self.current_directory_level = self.current_directory_level[part]

            # Process files with matching extensions.
            for file in files:
                if (
                    file in self.exclude_files
                    or file.startswith(".")
                    or file.startswith("__")
                ):
                    continue

                self.filename = file
                if self._matches_extension(file):
                    logger.info(f"Reading file: {file}")
                    self.filepath = Path(project_root) / file
                    file_payload: str = self.read_file_content()
                    self.current_directory_level[file] = file_payload

        return self.directory_tree

    def _matches_extension(self, filename: str) -> bool:
        """
        Check if file matches any of the include extensions.

        Args:
            filename: The filename to check.

        Returns:
            True if matches or no extensions specified.
        """
        if not self.include_extensions:
            return True

        return any(filename.endswith(ext) for ext in self.include_extensions)

    def get_files_by_extensions(
        self,
        directory_path: Optional[str] = None,
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Retrieve a list of file paths in a directory that match specified extensions.

        Traverses the given directory recursively, collecting paths of files with matching
        extensions while excluding specified directories and files. Useful for targeted
        file operations without reading content.

        Args:
            directory_path: The directory to search (absolute or relative). Defaults to
                            current working directory.
            extensions: List of file extensions to filter by (e.g., ['.py', '.txt']).
                        Defaults to self.include_extensions if None.

        Returns:
            List of Path objects for matching files.

        """
        extensions_to_use: List[str]
        if extensions is not None:
            self.include_extensions = extensions
            extensions_to_use = extensions
        else:
            extensions_to_use = self.include_extensions

        if not extensions_to_use:
            logger.warning("No extensions provided; returning empty list.")
            return []

        search_dir = (
            Path(directory_path) if directory_path else self.current_working_directory
        )
        if not search_dir.is_dir():
            logger.warning(
                f"Directory does not exist or is not a directory: {search_dir}"
            )
            return []

        matching_files: List[Path] = []
        file_count: int = 0

        try:
            for item in search_dir.rglob("*"):
                if item.is_file() and self._matches_extension(item.name):
                    # Skip files in excluded directories.
                    if any(
                        excluded in item.parts for excluded in self.exclude_directories
                    ):
                        continue

                    matching_files.append(item)
                    file_count += 1

            logger.info(f"Found {file_count} matching files in {search_dir}")
            return matching_files

        except Exception as e:
            logger.error(f"Error searching directory {search_dir}: {e}")
            return []

    def _read_file_content_helper(self, file_path: Path) -> str:
        """
        Helper method to read file content, respecting max_file_bytes limit.

        Args:
            file_path: The path to the file.

        Returns:
            File content as string, or empty if skipped.
        """
        try:
            file_size: int = file_path.stat().st_size

            # Skip files larger than max_file_bytes.
            if self.max_file_bytes and file_size > self.max_file_bytes:
                logger.warning(
                    f"Skipping file {file_path} - size {file_size} exceeds max {self.max_file_bytes} bytes"
                )
                return ""

            # Try reading as text.
            with open(file_path, encoding=self.encoding) as file:
                file_payload: str = file.read()
                return file_payload

        except UnicodeDecodeError:
            # Handle binary files.
            logger.warning(f"Skipping binary file: {file_path}")
            return ""
        except Exception as reading_error:
            logger.error(f"Error reading file {file_path}: {reading_error}")
            return ""

    def read_file_content(self) -> str:
        """
        Read file content, respecting max_file_bytes limit.

        Returns:
            File content as string.
        """
        return self._read_file_content_helper(self.filepath)

    def read_file_content_for_path(self, file_path: Path) -> str:
        """
        Read file content for a specific path, similar to read_file_content but without setting self.filepath.

        Used for process_directory to avoid side effects.

        Args:
            file_path: The path to the file.

        Returns:
            File content as string.
        """
        return self._read_file_content_helper(file_path)

    def write_file(
        self,
        filepath: str,
        payload: str,
        encoding: str = "utf-8",
        backup: bool = False,
        atomic: bool = True,
    ) -> bool:
        """
        Write content to file with optional backup and atomic write.

        Args:
            filepath: The file path to write to.
            payload: The content to write.
            encoding: File encoding.
            backup: Whether to create a backup.
            atomic: Whether to use atomic write.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.filepath = Path(filepath)
            self.filename = self.filepath.name
            self.payload = payload
            self.encoding = encoding
            self.backup = backup
            self.atomic = atomic

            # Create parent directories if they don't exist.
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {self.filepath.parent}")

            # Create backup if requested and file exists.
            if self.backup and self.filepath.exists():
                self.backup_path = self.filepath.with_suffix(
                    self.filepath.suffix + ".bak"
                )
                shutil.copy2(self.filepath, self.backup_path)
                logger.info(f"Created backup: {self.backup_path}")

            if self.atomic:
                # Atomic write using temporary file.
                self._atomic_write()
            else:
                # Standard write.
                with open(self.filepath, "w", encoding=self.encoding) as file:
                    file.write(self.payload)

            logger.info(
                f"Successfully wrote {len(self.payload)} characters to {self.filename}"
            )
            return True

        except PermissionError as permission_error:
            logger.error(
                f"Permission denied writing to {self.filename}: {permission_error}"
            )
            return False
        except OSError as os_error:
            logger.error(f"OS error writing to {self.filename}: {os_error}")
            return False
        except Exception as exception:
            logger.error(
                f"Unexpected error writing to {self.filename}: {exception}",
                exc_info=True,
            )
            return False

    def _atomic_write(self) -> None:
        """
        Perform atomic write operation to prevent data corruption.

        Uses temporary file in same directory, then atomic replace operation.
        """
        # Create temporary file in same directory for atomic move.
        with NamedTemporaryFile(
            mode="w",
            encoding=self.encoding,
            dir=self.filepath.parent,
            delete=False,
            prefix=f".tmp_{self.filepath.name}_",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path: Path = Path(temporary_file.name)

            try:
                # Write content to temporary file.
                temporary_file.write(self.payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())  # Ensure data is written to disk.

                # Copy permissions from original file if it exists.
                if self.filepath.exists():
                    temporary_path.chmod(self.filepath.stat().st_mode)

            except Exception:
                # Clean up temp file on error.
                temporary_path.unlink(missing_ok=True)
                raise

            # Atomic replace operation.
            try:
                os.replace(temporary_path, self.filepath)
                logger.info(
                    f"Successfully wrote {len(self.payload)} characters to {self.filename}"
                )

            except Exception:
                # Clean up temp file if replace fails.
                temporary_path.unlink(missing_ok=True)
                raise

    def get_git_info(self) -> Dict[str, Optional[str]]:
        """
        Get git username and remote URL.

        Returns:
            Dictionary with 'username' and 'url' keys.
        """
        git_info: Dict[str, Optional[str]] = {"username": None, "url": None}

        try:
            # Get username.
            result: subprocess.CompletedProcess = subprocess.run(
                ["git", "config", "user.name"],
                cwd=str(self.project_root_path),
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                git_info["username"] = result.stdout.strip()

            # Get remote URL.
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=str(self.project_root_path),
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                git_info["url"] = result.stdout.strip()

            return git_info

        except FileNotFoundError:
            logger.error(
                "Git command not found. Please ensure git is installed and in your PATH."
            )
            raise
        except Exception as e:
            logger.error(f"Error getting git info: {e}")
            return git_info

    @staticmethod
    def cleanup_escapes(input_str: str) -> str:
        """
        Clean up escaped backslashes and newlines (e.g., \\n -> \n) in a string
        using unicode_escape decoding.

        Args:
            input_str: The input string with escaped sequences.

        Returns:
            The cleaned string with interpreted escapes.
        """
        try:
            # Decode escaped sequences like \\n to \n, \\t to \t, etc.
            cleaned: str = codecs.decode(input_str, "unicode_escape")
            return cleaned

        except UnicodeDecodeError:
            # Fallback for invalid escapes: return original.
            logger.warning(f"Could not decode unicode escapes in string: {input_str}")
            return input_str

    def create_patch(self) -> Any:
        """
        Creates a git diff patch.

        Returns:
            The git diff patch as a string.

        Raises:
            subprocess.CalledProcessError: If the git command fails.
        """
        logger.debug(f"Git diff command: {self.git_diff_command}")

        result: subprocess.CompletedProcess = subprocess.run(
            self.git_diff_command,
            cwd=str(self.current_working_directory),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
