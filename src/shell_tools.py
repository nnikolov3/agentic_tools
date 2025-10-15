# File: src/shell_tools.py
"""
Safe filesystem helpers for assembling LLM context.

Responsibilities:
- Collect recently modified source files with allowlists, excludes, and byte caps.
- Load designated documentation files if present.
- Discover documentation by patterns and semantic signals when filenames drift.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


def _is_excluded_dir(path: Path, exclude_dirs: Tuple[str, ...]) -> bool:
    names = {p.lower() for p in path.resolve().parts}
    return any(ex.lower() in names for ex in exclude_dirs)


def _read_head_utf8(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except OSError as e:
        logger.error("Failed to read %s: %s", path, e)
        return ""


def _within_time(stat_mtime: float, cutoff_epoch: float) -> bool:
    return stat_mtime >= cutoff_epoch


def collect_recent_sources(
    project_root: str,
    include_extensions: Tuple[str, ...],
    exclude_dirs: Tuple[str, ...],
    recent_minutes: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> str:
    """
    Collect recently modified allowed files up to max_total_bytes with boundaries.
    """
    now = time.time()
    cutoff = now - (recent_minutes * 60)
    root = Path(project_root).resolve()
    if not root.exists():
        logger.warning("Project root does not exist: %s", root)
        return ""

    total = 0
    chunks: List[str] = []
    allow = {e.lower() for e in include_extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        if _is_excluded_dir(cur, exclude_dirs):
            dirnames[:] = []
            continue
        for name in filenames:
            p = cur / name
            if p.suffix.lower() not in allow:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if not _within_time(st.st_mtime, cutoff) or st.st_size <= 0:
                continue
            head = _read_head_utf8(p, max_file_bytes)
            header = f"\n===== FILE: {p.relative_to(root)} | SIZE: {st.st_size} bytes =====\n"
            piece = header + head + "\n"
            new_total = total + len(piece.encode("utf-8", errors="ignore"))
            if new_total > max_total_bytes:
                logger.info("Source payload limit reached (%d bytes)", max_total_bytes)
                return "".join(chunks)
            chunks.append(piece)
            total = new_total
    return "".join(chunks)


def _glob_candidates(root: Path, patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    if not patterns:
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        for name in filenames:
            lname = name.lower()
            for pat in patterns:
                if fnmatch.fnmatch(lname, pat.lower()):
                    out.append(cur / name)
                    break
    # De-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _scan_markdown(root: Path, exclude_dirs: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        if _is_excluded_dir(cur, exclude_dirs):
            dirnames[:] = []
            continue
        for name in filenames:
            p = cur / name
            if p.suffix.lower() == ".md":
                out.append(p)
    return out


def _score_by_signals(text: str, keywords: Sequence[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t)


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
        for p in candidates:
            head = _read_head_utf8(p, max_doc_bytes)
            score = _score_by_signals(head, keywords)
            if score > best_score:
                best_score = score
                best_path = p
        if best_path:
            selected.append(best_path)

    # Fallback: if still empty, take first few markdowns
    if not selected:
        selected = candidates[:max_docs]
    else:
        selected = selected[:max_docs]

    result: List[Tuple[Path, str]] = []
    for p in selected:
        content = _read_head_utf8(p, max_doc_bytes)
        result.append((p, content))
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
    for rel in docs_paths:
        p = (root / rel).resolve()
        if p.exists() and p.is_file():
            out.append((p, _read_head_utf8(p, max_doc_bytes)))
    return out
