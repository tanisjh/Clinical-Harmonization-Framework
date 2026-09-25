"""Convenience source readers for trace tooling.

These readers inspect current files for interactive tracing. They are not
strict-replay authority; replay verifies pinned manifests and SourceBindings
through its own reader path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cidc_io import read_cidc_csv


def read_source(project_root, resolved_path: str, sheet: str = "") -> pd.DataFrame:
    """Open a trace source, dispatching by explicit sheet or file suffix."""
    path = Path(project_root) / resolved_path
    if sheet:
        return pd.read_excel(path, sheet_name=sheet).reset_index(drop=True)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path).reset_index(drop=True)
    return read_cidc_csv(path).df.reset_index(drop=True)
