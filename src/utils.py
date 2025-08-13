from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Console handler that writes via tqdm.write() to avoid breaking bars."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write the formatted log record via tqdm to avoid breaking progress bars.

        Args:
            record (logging.LogRecord): The log record to emit.

        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def get_logger(
    name: str, level: str = "INFO", enable: bool = True
) -> Optional[logging.Logger]:
    """Create a logger instance with a tqdm-friendly console handler.

    Args:
        name (str): Logger name.
        level (str, optional): Logging level (case-insensitive). Defaults to "INFO".
        enable (bool, optional): Whether to return a logger or None. Defaults to True.

    Returns:
        Optional[logging.Logger]: Configured logger, or None if `enable` is False.
    """
    if not enable:
        return None
    logger = logging.getLogger(name)
    logger.propagate = False  # prevent duplicate logs via root

    # remove any existing handlers once, to avoid duplicates in notebooks
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # console handler that plays nice with tqdm
    ch = TqdmLoggingHandler()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    logger.setLevel(level.upper())
    return logger


def prepare_run_dir(base_path: Path) -> Path:
    """Create a timestamped run directory under the given base path.

    Args:
        base_path (Path): Directory under which the new run directory will be created.

    Returns:
        Path: The created run directory path.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base_path / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Save a small JSON file with pretty formatting.

    Args:
        path (Path): File path to write.
        obj (Dict[str, Any]): Dictionary to save as JSON.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Union[Dict[str, Any], list, str, int, float, bool, None]:
    """Load a small JSON file.

    Args:
        path (Path): File path to read.

    Returns:
        Union[dict, list, str, int, float, bool, None]: Parsed JSON content.

    """
    with open(path, "r") as f:
        return json.load(f)
