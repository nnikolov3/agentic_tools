#!/usr/bin/env python3
"""
linter.py - A script for comprehensive, concurrent linting of Python files.

This script automates the process of quality enforcement by recursively finding
all Python files in a project and running mypy, ruff, and bandit against each
one. It leverages asyncio to run the linters concurrently for each file,
improving efficiency. The final report is printed to the console and can be
optionally saved to a file.

Why this design:
- Single Responsibility: Each function has a clearly defined purpose.
- Asynchronous Execution: Speeds up the linting process for multiple files.
- Robustness: Gracefully handles missing linter executables.
- Explicitness: Provides clear, deduplicated output and optional file saving.
"""

import argparse
import asyncio
import logging
import subprocess  # nosec B404
from pathlib import Path
from typing import Dict, List, Set

# --- Configuration Constants ---
EXCLUDE_DIRS: Set[str] = {
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs",
    ".git",
}

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_python_files(start_dir: Path) -> List[Path]:
    """
    Recursively finds all Python files in a directory, excluding specified folders.

    Why: Isolates file discovery logic for easy modification and type safety.
    """
    logger.info("Searching for Python files in: %s", start_dir)
    python_files: List[Path] = []
    for path in start_dir.rglob("*.py"):
        if not any(part in EXCLUDE_DIRS for part in path.parts):
            python_files.append(path)
    logger.info("Found %d Python files to lint.", len(python_files))
    return python_files


async def run_linter(command: List[str]) -> str:
    """
    Asynchronously runs a linter command and captures its output.

    Why: Uses non-blocking execution for efficiency and gracefully handles
    missing executables to prevent silent crashes, ensuring the user always
    receives feedback.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return stdout.decode("utf-8", "ignore") + stderr.decode("utf-8", "ignore")
    except FileNotFoundError:
        return f"Error: Linter executable not found: '{command[0]}'. Please ensure it is installed."


def process_output(output: str) -> List[str]:
    """
    Cleans and deduplicates linter output.

    Why: Standardizes linter output for clarity and consistency by removing
    duplicates and stripping whitespace.
    """
    unique_lines: Set[str] = set()
    for line in output.strip().split("\n"):
        stripped_line = line.strip()
        if stripped_line:
            unique_lines.add(stripped_line)
    return sorted(list(unique_lines))


def format_report(all_issues: Dict[Path, Dict[str, List[str]]]) -> str:
    """
    Formats the collected linting issues into a single report string.

    Why: This function centralizes the report generation logic, making it
    easy to change the output format without altering the linting process.
    """
    if not all_issues:
        success_message = "=" * 80 + "\n"
        success_message += "🎉 No linting issues found. All files are clean! 🎉\n"
        success_message += "=" * 80
        return success_message

    report_parts: List[str] = [
        "=" * 80,
        "🚨 Linting issues found. Please review the findings below. 🚨",
        "=" * 80,
    ]
    for file_path, issues_by_linter in all_issues.items():
        report_parts.append(f"\n\n{'─' * 25} FILE: {file_path} {'─' * 25}")
        for linter_name, issues in issues_by_linter.items():
            report_parts.append(f"\n  [{linter_name.upper()}]")
            for issue in issues:
                report_parts.append(f"    - {issue}")
    report_parts.append("\n" + "=" * 80)
    return "\n".join(report_parts)


async def main() -> None:
    """
    The main entry point for the script.

    Orchestrates file discovery, concurrent linting, and organized reporting.
    """
    parser = argparse.ArgumentParser(
        description="Run linters on Python files in the project."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save the linting report.",
        default=None,
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    python_files = find_python_files(project_root)

    all_issues: Dict[Path, Dict[str, List[str]]] = {}

    for file_path in python_files:
        linters_to_run = {
            "mypy": ["mypy", str(file_path)],
            "ruff": ["ruff", "check", str(file_path)],
            "bandit": ["bandit", str(file_path)],
        }

        tasks = {
            linter: run_linter(command) for linter, command in linters_to_run.items()
        }
        results = await asyncio.gather(*tasks.values())

        file_issues: Dict[str, List[str]] = {}
        for (linter_name, _), raw_output in zip(tasks.items(), results):
            processed_lines = process_output(raw_output)
            if processed_lines:
                file_issues[linter_name] = processed_lines

        if file_issues:
            all_issues[file_path] = file_issues

    report = format_report(all_issues)
    print(report)

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Linting report saved to: %s", args.output_file)
        except IOError as e:
            logger.error("Failed to write report to file: %s", e)

    if all_issues:
        exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Linting process interrupted by user.")
        exit(1)
