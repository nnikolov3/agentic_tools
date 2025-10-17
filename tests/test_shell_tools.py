"""
Tests for shell_tools utilities focusing on file head reading and project structure filtering.
"""

from __future__ import annotations

from pathlib import Path

from src.shell_tools import get_project_structure, read_file_head


def test_read_file_head_truncates_and_decodes(tmp_path: Path) -> None:
    """
    Ensure read_file_head returns a UTF-8 decoded string with replacement on invalid bytes.
    """
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"\xffabcdef")

    content = read_file_head(file_path, max_bytes=3)

    assert content == "�ab"


def test_get_project_structure_respects_exclusions(tmp_path: Path) -> None:
    """
    Verify that get_project_structure omits directories passed via the exclude_entries parameter.
    """
    included_directory = tmp_path / "included"
    included_directory.mkdir()
    (included_directory / "kept.txt").write_text("keep me", encoding="utf-8")

    excluded_directory = tmp_path / "excluded"
    excluded_directory.mkdir()
    (excluded_directory / "drop.txt").write_text("drop me", encoding="utf-8")

    structure = get_project_structure(
        project_root=str(tmp_path),
        max_depth=2,
        exclude_entries=("excluded",),
    )

    assert "excluded" not in structure
    assert "included/" in structure
