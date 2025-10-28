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
    def __init__(self, agent: str, config: Dict[str, Any], target_directory: Path) -> None:
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
        self.target_directory: Path = target_directory

    def get_project_file_tree(self) -> Dict[str, Any>:
# ...
    def get_design_docs_content(self) -> str:
        design_docs_content_parts: List[str> = []
        for doc_path_str in self.design_docs:
            full_path = self.target_directory / doc_path_str
            if full_path.is_file():
# ...
    def read_file_content(self, file_path: Path) -> str:
        try:
            file_path = self.target_directory / file_path
            with file_path.open(encoding=self.encoding) as file_handle:
# ...
    def get_git_info(self) -> Dict[str, Optional[str]>:
        git_info: Dict[str, Optional[str]> = {"username": None, "url": None}
        try:
            username_result = subprocess.run(
                [self.git_executable, "config", "user.name"],
                cwd=str(self.target_directory),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if username_result.returncode == 0:
                git_info["username"] = username_result.stdout.strip()

            url_result = subprocess.run(
                [self.git_executable, "config", "--get", "remote.origin.url"],
                cwd=str(self.target_directory),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if url_result.returncode == 0:
                git_info["url"] = url_result.stdout.strip()

            return git_info
# ...
    def create_patch(self) -> str:
        command = [self.git_executable, *self.git_diff_command]
        logger.debug("Running git diff command: %s", ' '.join(command))
        try:
            result = subprocess.run(
                command,
                cwd=str(self.target_directory),
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
