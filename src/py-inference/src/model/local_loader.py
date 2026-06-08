"""Local model loader utilities."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_local_model(model_path: str, search_dirs: list[str] | None = None) -> str | None:
    """Find a model in local directories.

    Args:
        model_path: Model identifier or path.
        search_dirs: Additional directories to search.

    Returns:
        Absolute path to model directory if found, None otherwise.
    """
    # Direct path
    p = Path(model_path)
    if p.exists():
        return str(p.resolve())

    # Search in common locations
    dirs = search_dirs or []
    dirs.extend([
        "/models",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    ])

    for d in dirs:
        candidate = Path(d) / model_path.replace("/", "--")
        if candidate.exists():
            logger.info("found model locally", extra={"path": str(candidate)})
            return str(candidate)

    return None
