import os
import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
import subprocess
import codecs


logger = logging.getLogger(__name__)


class ShellTools:
    def __init__(self, agent, config: dict):
        self.config = config
        self.agent_config = self.config.get(agent)
        if self.agent_config is None:
            raise ValueError(f"Agent '{agent}' not found in configuration.")
        self.agent_prompt = self.agent_config.get("prompt")
        self.agent_model_name = self.agent_config.get("model_name")
        self.agent_temperature = self.agent_config.get("temperature")
        self.agent_description = self.agent_config.get("description")
        self.agent_model_provider = self.agent_config.get("model_provider")
        self.agent_alternative_model = self.agent_config.get("alternative_model")
        self.agent_alternative_provider = self.agent_config.get("alternative_provider")
        self.project_root = config.get("project_root")
        self.agent_skills = self.agent_config.get("skills")
        self.design_docs = config.get("design_docs")
        self.source = config.get("source")
        self.project_directories = config.get("project_directories")
        self.include_extensions = config.get("include_extensions")
        self.max_file_bytes = config.get("max_file_bytes")
        self.exclude_directories = config.get("exclude_directories", [])
        self.exclude_files = config.get("exclude_files", [])
        self.recent_minutes = config.get("recent_minutes")
        self.filepath = Path()
        self.filename = ""
        self.payload = ""
        self.encoding = "utf-8"
        self.backup = False
        self.atomic = True
        self.backup_path = Path()
        self.root_path: Path | None = None
        self.directory_tree: dict = {}
        self.current_directory_level: dict = {}
        self.project_root_path = Path()
        self.directories: list = []
        self.files: list = []
        self.relative_path: Path | None = None

    def concatenate_all_files(self):
        """Traverse the project directories and concatenate all files."""
        concatenated_files_dict = {}

        # Process all project directories
        if self.project_directories:
            concatenated_files_dict.update(
                self._process_directories(self.project_directories, "project")
            )

        return concatenated_files_dict

    def _process_directories(self, directories: list, directory_type: str) -> dict:
        """
        Helper method to process a list of directories and build their file trees.

        Args:
            directories: List of directory paths to process
            directory_type: Type label for logging ("source" or "project")

        Returns:
            Dictionary containing concatenated file contents
        """
        result = {}

        for directory in directories:
            logger.info(f"Processing {directory_type} directory: {directory}")
            print(f"Processing {directory_type} directory: {directory}")

            self.project_root_path = Path(directory)

            if not self.project_root_path.exists():
                logger.warning(
                    f"{directory_type.capitalize()} directory does not exist: {directory}"
                )
                continue

            result.update(self._build_directory_tree())

        return result

    def _build_directory_tree(self) -> dict:
        """Build nested dictionary structure for a directory tree."""
        self.directory_tree = {}

        for project_root, self.directories, self.files in os.walk(
            self.project_root_path
        ):
            # Exclude directories in-place to prevent os.walk from descending into them
            self.directories[:] = [
                directory
                for directory in self.directories
                if directory not in self.exclude_directories
            ]

            # Calculate relative path from root
            self.relative_path = Path(project_root).relative_to(self.project_root_path)

            # Navigate to correct position in nested dict
            self.current_directory_level = self.directory_tree
            if self.relative_path and str(self.relative_path) != ".":
                for part in self.relative_path.parts:
                    if part not in self.current_directory_level:
                        self.current_directory_level[part] = {}
                    self.current_directory_level = self.current_directory_level[part]

            # Process files with matching extensions
            for file in self.files:
                if file in self.exclude_files:
                    continue

                self.filename = file
                if self._matches_extension():
                    self.filepath = Path(project_root) / file
                    file_payload = self.read_file_content()
                    self.current_directory_level[file] = file_payload

        return self.directory_tree

    def _matches_extension(self) -> bool:
        """Check if file matches any of the include extensions."""
        if not self.include_extensions:
            return True
        return any(self.filename.endswith(ext) for ext in self.include_extensions)

    def read_file_content(self) -> str:
        """Read file content, respecting max_file_bytes limit."""
        try:
            file_path = Path(self.filepath)
            file_size = file_path.stat().st_size

            # Skip files larger than max_file_bytes
            if self.max_file_bytes and file_size > self.max_file_bytes:
                logger.warning(
                    f"Skipping file {self.filepath} - size {file_size} exceeds max {self.max_file_bytes} bytes"
                )
                return f"<File too large: {file_size} bytes>"

            # Try reading as text
            with open(file_path, "r", encoding=self.encoding) as file:
                file_payload = file.read()
            return file_payload

        except UnicodeDecodeError:
            # Handle binary files
            logger.warning(f"Skipping binary file: {self.filepath}")
            return "<Binary file>"

        except Exception as readingError:
            logger.error(f"Error reading file {self.filepath}: {readingError}")
            return f"<Error reading file: {readingError}>"

    def write_file(
        self,
        filepath,
        payload,
        encoding: str = "utf-8",
        backup: bool = False,
        atomic: bool = True,
    ) -> bool:
        """Write content to file with optional backup and atomic write."""
        try:
            self.filepath = Path(filepath)
            self.filename = self.filepath.name
            self.payload = payload
            self.encoding = encoding
            self.backup = backup
            self.atomic = atomic

            # Create parent directories if they don't exist
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {self.filepath.parent}")

            # Create backup if requested and file exists
            if self.backup and self.filepath.exists():
                self.backup_path = self.filepath.with_suffix(
                    self.filepath.suffix + ".bak"
                )
                shutil.copy2(self.filepath, self.backup_path)
                logger.info(f"Created backup: {self.backup_path}")

            if self.atomic:
                # Atomic write using temporary file
                self._atomic_write()
            else:
                # Standard write
                with open(self.filepath, "w", encoding=self.encoding) as file:
                    file.write(self.payload)
                logger.info(
                    f"Successfully wrote {len(self.payload)} characters to {self.filename}"
                )

            return True

        except PermissionError as permissionError:
            logger.error(
                f"Permission denied writing to {self.filename}: {permissionError}"
            )
            return False

        except OSError as osError:
            logger.error(f"OS error writing to {self.filename}: {osError}")
            return False

        except Exception as _Exception:
            logger.error(
                f"Unexpected error writing to {self.filename}: {_Exception}",
                exc_info=True,
            )
            return False

    def _atomic_write(self):
        """
        Perform atomic write operation to prevent data corruption.
        Uses temporary file in same directory, then atomic replace operation.
        """
        # Create temporary file in same directory for atomic move
        with NamedTemporaryFile(
            mode="w",
            encoding=self.encoding,
            dir=self.filepath.parent,
            delete=False,
            prefix=f".tmp_{self.filepath.name}_",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            try:
                # Write content to temporary file
                temporary_file.write(self.payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())  # Ensure data is written to disk

                # Copy permissions from original file if it exists
                if self.filepath.exists():
                    temporary_path.chmod(self.filepath.stat().st_mode)

            except Exception:
                # Clean up temp file on error
                temporary_path.unlink(missing_ok=True)
                raise

        # Atomic replace operation
        try:
            os.replace(temporary_path, self.filepath)
            logger.info(
                f"Successfully wrote {len(self.payload)} characters to {self.filename}"
            )
        except Exception:
            # Clean up temp file if replace fails
            temporary_path.unlink(missing_ok=True)
            raise

    def get_git_info(self) -> dict:
        """Get git username and remote URL."""

        git_info: dict[str, str | None] = {"username": None, "url": None}

        try:
            # Get username
            result = subprocess.run(
                ["git", "config", "user.name"],
                cwd=self.project_root if self.project_root else ".",
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_info["username"] = result.stdout.strip()

            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=self.project_root if self.project_root else ".",
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_info["url"] = result.stdout.strip()

            return git_info

        except FileNotFoundError:
            logger.error("Git command not found. Please ensure git is installed and in your PATH.")
            raise

        except Exception as e:
            logger.error(f"Error getting git info: {e}")
            return git_info

    def cleanup_escapes(self, input_str):
        """
        Clean up escaped backslashes and newlines (e.g., \\n -> \n) in a string
        using unicode_escape decoding.

        Args:
            input_str (str): The input string with escaped sequences.

        Returns:
            str: The cleaned string with interpreted escapes.
        """
        try:
            # Decode escaped sequences like \\n to \n, \\t to \t, etc.
            cleaned = codecs.decode(input_str, "unicode_escape")
            return cleaned
        except UnicodeDecodeError:
            # Fallback for invalid escapes: return original
            logger.warning(f"Could not decode unicode escapes in string: {input_str}")
            return input_str
