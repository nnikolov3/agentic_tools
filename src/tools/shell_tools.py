# src/tools/shell_tools.py
import asyncio
import codecs
import logging
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional

import httpx
from bs4 import BeautifulSoup

logger: logging.Logger = logging.getLogger(__name__)


class ShellTools:
    def __init__(self, agent: str, config: Dict[str, Any], target_directory: Optional[Path]) -> None:
        self.config: Dict[str, Any] = config
        self.agent_config: Dict[str, Any] = self.config.get(agent, {})

        self.git_diff_command: List[str] = self.config.get(
            "git_diff_command",
            [
                "diff",
                "HEAD",
                "--patch-with-raw",
                "--minimal",
                "--patience",
                ":!uv.lock",
                ":!MANIFEST.in",
            ],
        )
        git_executable_path: Optional[str] = shutil.which("git")
        if not git_executable_path:
            logger.error(
                "Git command not found. Please ensure git is installed and in your PATH.",
            )
            raise FileNotFoundError("Git executable not found.")
        self.git_executable: str = git_executable_path

        self.sync_executable: Optional[str] = shutil.which("sync")
        if not self.sync_executable:
            logger.warning("'sync' command not found. File system sync may be skipped.")

        self.design_docs: List[str] = config.get("design_docs", [])
        self.project_directories: List[str] = config.get("project_directories", [])
        self.include_extensions: List[str] = config.get("include_extensions", [])
        self.exclude_directories: List[str] = config.get("exclude_directories", [])
        self.exclude_files: List[str] = config.get("exclude_files", [])
        self.encoding: str = "utf-8"

        if target_directory is not None:
            self.current_working_directory = target_directory
        else :
            self.current_working_directory = Path.cwd()


    def get_project_file_tree(self) -> Dict[str, Any]:
        project_tree: Dict[str, Any] = {}
        for directory_str in self.project_directories:
            directory_path = Path(directory_str)
            logger.info("Processing project directory: %s", directory_path)

            if not directory_path.is_dir():
                logger.warning(
                    "Project directory does not exist or is not a directory: %s",
                    directory_path,
                )
                continue

            project_tree.update(self._build_directory_tree(directory_path))

        return project_tree

    def _build_directory_tree(self, root_path: Path) -> Dict[str, Any]:
        directory_tree: Dict[str, Any] = {}
        for file_path in self._get_filtered_files(root_path, self.include_extensions):
            relative_path = file_path.relative_to(root_path)

            current_level = directory_tree
            for part in relative_path.parts[:-1]:
                current_level = current_level.setdefault(part, {})

            file_name = relative_path.name
            logger.info("Reading file: %s", file_path)
            file_content = self.read_file_content(file_path)
            if file_content:
                current_level[file_name] = file_content

        return directory_tree

    def _should_exclude_file(self, file_name: str) -> bool:
        return (
            file_name in self.exclude_files
            or file_name.startswith(".")
            or file_name.startswith("__")
        )

    def process_directory(self, directory_path: Path) -> str:
        if not directory_path.is_dir():
            logger.warning(
                "Directory does not exist or is not a directory: %s",
                directory_path,
            )
            return ""

        concatenated_content_parts: List[str] = []
        file_count = 0

        try:
            for file_path in self._get_filtered_files(
                directory_path,
                self.include_extensions,
            ):
                file_content = self.read_file_content(file_path)
                logger.info("Reading file: %s", file_path)

                if not file_content or file_content.strip().startswith("<"):
                    continue

                relative_path = file_path.relative_to(directory_path)
                header = f"\n\n--- File: {relative_path} ---\n\n"
                concatenated_content_parts.append(header + file_content)
                file_count += 1

            if file_count == 0:
                logger.info("No matching files found in directory: %s", directory_path)

            logger.info("Concatenated %d files from %s", file_count, directory_path)
            return "".join(concatenated_content_parts).strip()

        except OSError as error:
            logger.error(
                f"Error processing directory {directory_path}: {error}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to process directory {directory_path}",
            ) from error

    def get_design_docs_content(self) -> str:
        design_docs_content_parts: List[str] = []
        for doc_path_str in self.design_docs:
            full_path = self.current_working_directory / doc_path_str
            if full_path.is_file():
                content = self.read_file_content(full_path)
                if content:
                    design_docs_content_parts.append(content)
            else:
                logger.warning(
                    f"Design document not found or is not a file: {full_path}",
                )

        return "\n\n".join(design_docs_content_parts)

    def _matches_extension(self, filename: str) -> bool:
        if not self.include_extensions:
            return True
        return any(filename.endswith(ext) for ext in self.include_extensions)

    def _is_in_excluded_directory(self, file_path: Path) -> bool:
        return any(
            ((excluded in file_path.parts) for excluded in self.exclude_directories),
        )

    def _get_filtered_files(
        self,
        directory_path: Path,
        extensions: List[str],
    ) -> Generator[Path, None, None]:
        for item in directory_path.rglob("*"):
            if (
                item.is_file()
                and (
                    not extensions or any(item.name.endswith(ext) for ext in extensions)
                )
                and not self._is_in_excluded_directory(item)
                and not self._should_exclude_file(item.name)
            ):
                yield item

    def get_files_by_extensions(
        self,
        directory_path: Path,
        extensions: List[str],
    ) -> List[Path]:
        if not directory_path.is_dir():
            logger.warning(
                f"Directory does not exist or is not a directory: {directory_path}",
            )
            return []

        try:
            matching_files = list(self._get_filtered_files(directory_path, extensions))
            logger.info(
                f"Found {len(matching_files)} matching files in {directory_path}",
            )
            return matching_files
        except OSError as error:
            logger.error(
                f"Error searching directory {directory_path}: {error}",
                exc_info=True,
            )
            return []

    def read_file_content(self, file_path: Path) -> str:
        try:
            file_path = self.current_working_directory / file_path
            with file_path.open(encoding=self.encoding) as file_handle:
                return file_handle.read()
        except UnicodeDecodeError:
            logger.warning(f"Skipping binary file (UnicodeDecodeError): {file_path}")
            return ""
        except Exception as reading_error:
            logger.error(f"Error reading file {file_path}: {reading_error}")
            return ""

    def write_file(
        self,
        filepath: Path,
        payload: str,
        backup: bool = False,
        atomic: bool = True,
    ) -> bool:
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {filepath.parent}")

            if backup and filepath.exists():
                backup_path = filepath.with_suffix(filepath.suffix + ".bak")
                shutil.copy2(filepath, backup_path)
                logger.info(f"Created backup: {backup_path}")

            if atomic:
                self._atomic_write(filepath, payload)
            else:
                with filepath.open("w", encoding=self.encoding) as file_handle:
                    file_handle.write(payload)

            logger.info(
                f"Successfully wrote {len(payload)} characters to {filepath.name}",
            )
            return True

        except (PermissionError, OSError) as file_error:
            logger.error(f"File system error writing to {filepath.name}: {file_error}")
            return False
        except Exception as exception:
            logger.error(
                f"Unexpected error writing to {filepath.name}: {exception}",
                exc_info=True,
            )
            return False

    def _atomic_write(self, target_path: Path, payload: str) -> None:
        temp_file_path: Optional[Path] = None
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
                if platform.system() != "Windows" and self.sync_executable:
                    try:
                        subprocess.run([self.sync_executable], check=True, capture_output=True)
                    except (
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                    ) as sync_error:
                        logger.warning(
                            f"Failed to execute '{self.sync_executable}' command: {sync_error}",
                        )
                elif platform.system() != "Windows":
                    logger.warning("'sync' executable not found, skipping file system sync.")

            if target_path.exists():
                shutil.copymode(target_path, temp_file_path)

            temp_file_path.rename(target_path)

        except OSError as error:
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise IOError(f"Atomic write to {target_path} failed") from error

    async def fetch_urls_content(self, urls: List[str]) -> str:
        """
        Fetches content from a list of URLs concurrently and extracts clean text.

        Args:
            urls: A list of URL strings to fetch.

        Returns:
            A single string concatenating the clean text from all successfully fetched URLs.
        """
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [client.get(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        content_parts: List[str] = []
        for i, result in enumerate(results):
            url = urls[i]
            if isinstance(result, httpx.Response) and result.status_code == 200:
                header = f"\n\n--- Content from: {url} ---\n\n"

                # Parse HTML and extract text
                soup = BeautifulSoup(result.text, "html.parser")
                clean_text = soup.get_text(strip=True)
                content_parts.append(header + clean_text)
                logger.info("Successfully fetched and parsed content from %s", url)
            else:
                error_msg = (
                    result
                    if isinstance(result, Exception)
                    else f"Status code: {getattr(result, 'status_code', 'N/A')}"
                )
                logger.warning("Failed to fetch content from %s: %s", url, error_msg)

        return "\n".join(content_parts).strip()

    async def run_project_linters(self) -> str:
        """
        Runs all configured linters against the project and aggregates their output.
        """
        linter_configs: dict[str, list[str]] = self.config.get("linters", {})
        if not linter_configs:
            logger.warning(
                "No linters configured in agentic_tools.toml under [agentic-tools.linters]",
            )
            return "No linters configured."

        report_parts: list[str] = []
        for tool_name, command in linter_configs.items():
            executable = command[0]
            if not shutil.which(executable):
                error_msg = f"Linter '{tool_name}' not found. Please ensure '{executable}' is installed."
                logger.error(error_msg)
                report_parts.append(
                    f"--- {tool_name.upper()} FAILED ---\n{error_msg}\n"
                )
                continue

            logger.info("Running linter: %s", ' '.join(command))
            try:
                # Linters often use non-zero exit codes to indicate issues, so we don't check for success.
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                output = stdout.decode(self.encoding, errors="ignore") + stderr.decode(
                    self.encoding,
                    errors="ignore",
                )

                if output.strip():
                    report_parts.append(
                        f"--- Linter Report: {tool_name.upper()} ---\n{output.strip()}\n",
                    )

            except Exception as e:
                logger.error("Error running linter '%s': %s", tool_name, e)
                report_parts.append(f"--- {tool_name.upper()} FAILED ---\n{e}\n")

        if not report_parts:
            return "All linters passed successfully. No issues found."

        return "\n".join(report_parts)

    def get_git_info(self) -> Dict[str, Optional[str]]:
        git_info: Dict[str, Optional[str]] = {"username": None, "url": None}
        try:
            username_result = subprocess.run(
                [self.git_executable, "config", "user.name"],
                cwd=str(self.current_working_directory),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if username_result.returncode == 0:
                git_info["username"] = username_result.stdout.strip()

            url_result = subprocess.run(
                [self.git_executable, "config", "--get", "remote.origin.url"],
                cwd=str(self.current_working_directory),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if url_result.returncode == 0:
                git_info["url"] = url_result.stdout.strip()

            return git_info

        except subprocess.TimeoutExpired:
            logger.error("Git command timed out while getting repository info.")
            return git_info
        except Exception as error:
            logger.error("Error getting git info: %s", error, exc_info=True)
            return git_info

    @staticmethod
    def cleanup_escapes(input_string: str) -> str:
        try:
            return codecs.decode(input_string, "unicode_escape")
        except UnicodeDecodeError:
            logger.warning(
                "Could not decode unicode escapes in string: %s...",
                input_string[:100],
            )
            return input_string

    def create_patch(self) -> str:
        command = [self.git_executable, *self.git_diff_command]
        logger.debug("Running git diff command: %s", ' '.join(command))
        try:
            result = subprocess.run(
                command,
                cwd=str(self.current_working_directory),
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            return result.stdout
        except subprocess.CalledProcessError as error:
            logger.error(
                "Git diff command failed with exit code %d:\nSTDOUT: %s\nSTDERR: %s",
                error.returncode,
                error.stdout,
                error.stderr,
            )
            raise RuntimeError("Failed to create git patch") from error
        except subprocess.TimeoutExpired as error:
            logger.error("Git diff command timed out: %s", error)
            raise RuntimeError("Git diff command timed out") from error
