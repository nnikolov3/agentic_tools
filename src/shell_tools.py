# File: src/shell_tools.py
"""
Safe filesystem helpers for assembling LLM context.

Responsibilities:
- Collect recently modified source files with allowlists, excludes, and byte caps.
- Load designated documentation files if present.
- Discover documentation by patterns and semantic signals when filenames drift.
- Gather project metadata including git information, project structure, and dependencies.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


def _is_excluded_dir(path: Path, exclude_dirs: Tuple[str, ...]) -> bool:
    names = {part.lower() for part in path.resolve().parts}
    return any(exclude_pattern.lower() in names for exclude_pattern in exclude_dirs)


def read_file_head(path: Path, max_bytes: int) -> str:
    """
    Read up to max_bytes from the given file and decode it as UTF-8 while replacing invalid bytes.
    """
    try:
        with path.open("rb") as file_handle:
            data = file_handle.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except OSError as read_error:
        logger.error("Failed to read %s: %s", path, read_error)
        return ""


def _within_time(stat_mtime: float, cutoff_epoch: float) -> bool:
    return stat_mtime >= cutoff_epoch


def _find_source_files(
    root: Path,
    include_extensions: Tuple[str, ...],
    exclude_dirs: Tuple[str, ...],
) -> List[Path]:
    """
    Finds all files matching the include_extensions and respecting exclude_dirs.
    """
    files: List[Path] = []
    allow = {extension.lower() for extension in include_extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        if _is_excluded_dir(current_dir, exclude_dirs):
            dirnames[:] = []
            continue
        for name in filenames:
            file_path = current_dir / name
            if file_path.suffix.lower() in allow:
                files.append(file_path)
    return files


def collect_recent_sources(
    project_root: str,
    include_extensions: Tuple[str, ...],
    exclude_dirs: Tuple[str, ...],
    recent_minutes: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> Tuple[str, List[Path]]:
    """
    Collect recently modified allowed files up to max_total_bytes with boundaries.
    Returns a tuple of (formatted_content, all_candidate_files).
    """
    now = time.time()
    cutoff = now - (recent_minutes * 60)
    root = Path(project_root).resolve()
    if not root.exists():
        logger.warning("Project root does not exist: %s", root)
        return "", []

    total = 0
    chunks: List[str] = []

    # Use the new generic function to find all candidate files
    candidate_files = _find_source_files(root, include_extensions, exclude_dirs)

    for file_path in candidate_files:
        try:
            stat_result = file_path.stat()
        except OSError:
            continue

        # Apply time filter
        if not _within_time(stat_result.st_mtime, cutoff) or stat_result.st_size <= 0:
            continue

        head = read_file_head(file_path, max_file_bytes)
        header = f"\n===== FILE: {file_path.relative_to(root)} | SIZE: {stat_result.st_size} bytes =====\n"
        piece = header + head + "\n"
        new_total = total + len(piece.encode("utf-8", errors="replace"))
        if new_total > max_total_bytes:
            logger.info("Source payload limit reached (%d bytes)", max_total_bytes)
            return "".join(chunks), candidate_files
        chunks.append(piece)
        total = new_total

    return "".join(chunks), candidate_files


def get_git_info(project_root: str) -> dict[str, str | bool | None]:
    """
    Get git repository information including URL, branch, and status.
    """
    root = Path(project_root).resolve()
    git_info: dict[str, str | bool | None] = {
        "url": None,
        "remote_url": None,
        "branch": None,
        "is_dirty": None,
    }

    try:
        # Get remote URL
        result = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()
            # Also set as primary URL if available
            git_info["url"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        logger.warning("Could not get git remote URL")

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip() or "HEAD (detached)"
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        logger.warning("Could not get git branch")

    try:
        # Check if repository is dirty
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            git_info["is_dirty"] = bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        logger.warning("Could not check git status")

    return git_info


def get_project_structure(
    project_root: str,
    max_depth: int = 3,
    exclude_entries: tuple[str, ...] | None = None,
) -> str:
    """
    Get a structured view of the project directory while respecting excluded names.
    """
    root = Path(project_root).resolve()
    structure_lines = [f"{root.name}/"]

    def _add_directory(
        path: Path, prefix: str, depth: int, excluded_names: set[str]
    ) -> None:
        if depth >= max_depth:
            return

        items = []
        for item in path.iterdir():
            if item.name.lower() in excluded_names:
                continue
            items.append(item)

        items.sort(key=lambda x: (x.is_file(), x.name.lower()))

        for item_index, item in enumerate(items):
            is_last = item_index == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "

            if item.is_dir():
                structure_lines.append(f"{prefix}{current_prefix}{item.name}/")
                _add_directory(item, prefix + next_prefix, depth + 1, excluded_names)
            else:
                structure_lines.append(f"{prefix}{current_prefix}{item.name}")

    try:
        exclusions = {name.lower() for name in exclude_entries or ()}
        _add_directory(root, "", 0, exclusions)
    except Exception as structure_error:
        logger.error("Error getting project structure: %s", structure_error)
        return f"Could not retrieve project structure: {structure_error}"

    return "\n".join(structure_lines)


def _glob_candidates(root: Path, patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    if not patterns:
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        for name in filenames:
            lname = name.lower()
            for pattern in patterns:
                if fnmatch.fnmatch(lname, pattern.lower()):
                    out.append(current_dir / name)
                    break
    # De-duplicate while preserving order using more idiomatic Python
    return list(dict.fromkeys(out))


def _scan_markdown(root: Path, exclude_dirs: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        if _is_excluded_dir(current_dir, exclude_dirs):
            dirnames[:] = []
            continue
        for name in filenames:
            file_path = current_dir / name
            if file_path.suffix.lower() == ".md":
                out.append(file_path)
    return out


def _score_by_signals(text: str, keywords: Sequence[str]) -> int:
    lower_text = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lower_text)


def discover_docs_and_load(
    project_root: str,
    exclude_dirs: Tuple[str, ...],
    patterns: Tuple[str, ...],
    signal_groups: Tuple[Tuple[str, Tuple[str, ...]], ...],
    max_docs: int,
    max_doc_bytes: int,
) -> List[Tuple[Path, str]]:
    """
    Find candidate markdown docs by patterns and semantic signals, then load tops.
    Returns list of (path, content_head).
    """
    root = Path(project_root).resolve()
    if not root.exists():
        return []

    candidates = _glob_candidates(root, patterns)
    if not candidates:
        candidates = _scan_markdown(root, exclude_dirs)

    # Group selection: pick top-1 doc per signal group
    selected: List[Path] = []
    for group_name, keywords in signal_groups:
        best_path: Path | None = None
        best_score = 0
        for candidate_path in candidates:
            head = read_file_head(candidate_path, max_doc_bytes)
            score = _score_by_signals(head, keywords)
            if score > best_score:
                best_score = score
                best_path = candidate_path
        if best_path:
            selected.append(best_path)

    # Fallback: if still empty, take first few markdowns
    if not selected:
        selected = candidates[:max_docs]
    else:
        selected = selected[:max_docs]

    result: List[Tuple[Path, str]] = []
    for selected_path in selected:
        content = read_file_head(selected_path, max_doc_bytes)
        result.append((selected_path, content))
    return result


def load_explicit_docs(
    project_root: str,
    docs_paths: Tuple[str, ...],
    max_doc_bytes: int,
) -> List[Tuple[Path, str]]:
    """
    Load explicit documentation files if they exist, returning (path, content_head).
    """
    root = Path(project_root).resolve()
    out: List[Tuple[Path, str]] = []
    for relative_path in docs_paths:
        file_path = (root / relative_path).resolve()
        if file_path.exists() and file_path.is_file():
            out.append((file_path, read_file_head(file_path, max_doc_bytes)))
    return out
