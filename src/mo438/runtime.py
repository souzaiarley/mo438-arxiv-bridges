"""Runtime helpers shared by every command: logging, stage timing, memory usage,
environment metadata, provenance blocks and JSON output."""

from __future__ import annotations

import json
import logging
import math
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__

LOGGER_NAME = "mo438"
_LIBRARIES = ("mo438", "numpy", "scipy", "pandas", "python-igraph", "networkx",
              "matplotlib", "PyYAML", "leidenalg", "scikit-learn")


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: Path | None, level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger to write to stderr and (optionally) ``log_file``."""
    logger = get_logger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def peak_memory_mb(children: bool = False) -> float | None:
    """Peak resident memory in MiB of this process, or (``children=True``) of the largest
    terminated child process such as a distance worker. ``None`` if unavailable."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    rss = resource.getrusage(who).ru_maxrss
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


class StageTimer:
    """Collects wall-clock durations of named pipeline stages."""

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block, logging its start, end, duration and peak memory."""
        logger = get_logger()
        logger.info("[stage] %s ...", name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.durations[name] = round(elapsed, 3)
            mem = peak_memory_mb()
            mem_txt = f", peak RSS {mem:.0f} MiB" if mem is not None else ""
            logger.info("[stage] %s done in %.2f s%s", name, elapsed, mem_txt)


def library_versions() -> dict[str, str]:
    """Versions of Python and of the libraries that influence the results."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for dist in _LIBRARIES:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = "not installed"
    return versions


# --------------------------------------------------------------------------- provenance


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                                cwd=Path(__file__).resolve().parent)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(command: str | None = None, **extra: Any) -> dict[str, Any]:
    """Who/what/when produced an output: git state, versions, command, timestamp."""
    status = _git("status", "--porcelain")
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mo438_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "command": command or " ".join(sys.argv),
        "environment": library_versions(),
        **extra,
    }


def _default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Pretty JSON (NaN -> null, numpy types converted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2, default=_default, ensure_ascii=False)
                    + "\n", encoding="utf-8")
    return path
