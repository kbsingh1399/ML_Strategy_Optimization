"""
Centralized data-path resolver.
Prefer local ``backtesting_data/`` (repo root); fall back to Google Drive only
if the local folder is missing or empty.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_DIR = _REPO_ROOT.parent / "Engine_1" / "backtesting_data"
_GDRIVE_DIR = Path(r"G:\My Drive\_Trading_Data\15m\parquet")


def resolve_pq_dir() -> str:
    """Return the preferred parquet data directory as a string path."""
    if Path("/content").exists() and any(Path("/content").glob("Master_*_15m_Final_Summary.parquet")):
        return "/content"
    if _GDRIVE_DIR.exists() and any(_GDRIVE_DIR.glob("Master_*_15m_Final_Summary.parquet")):
        return str(_GDRIVE_DIR)
    if _LOCAL_DIR.exists() and any(_LOCAL_DIR.iterdir()):
        return str(_LOCAL_DIR)
    return str(_GDRIVE_DIR)


PQ_DIR = resolve_pq_dir()


def summary_path(symbol: str, data_dir: str | None = None) -> str:
    d = data_dir or PQ_DIR
    primary = os.path.join(d, f"Master_{symbol}_15m_Final_Summary.parquet")
    if os.path.exists(primary):
        return primary
    alt = os.path.join(d, f"{symbol}_15m_summary.parquet")
    return alt if os.path.exists(alt) else primary


def footprint_path(symbol: str, data_dir: str | None = None) -> str:
    d = data_dir or PQ_DIR
    primary = os.path.join(d, f"Master_{symbol}_15m_Final_Footprint.parquet")
    if os.path.exists(primary):
        return primary
    alt = os.path.join(d, f"{symbol}_15m_footprint.parquet")
    return alt if os.path.exists(alt) else primary
