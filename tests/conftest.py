"""
Pytest configuration to ensure the project root is importable for package-style tests.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    """
    Add the repository root to sys.path if it is missing so imports of the src package succeed.
    """
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()
