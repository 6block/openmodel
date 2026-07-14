"""Model lifecycle manager."""

import logging
import shutil
from pathlib import Path

from .foc_client import FOCClient

logger = logging.getLogger(__name__)


def is_model_complete(path: Path) -> bool:
    """A cached model directory is usable only if it has a config AND at least one
    weight file (or a shard index). This guards against a half-written directory left
    by a crashed/partial download being treated as a valid cache, which would make
    vLLM fail to start forever (audit MEDIUM fix)."""
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.gguf", "*.index.json"):
        if any(path.glob(pattern)):
            return True
    return False


class ModelManager:
    """Manages model loading and caching."""

    def __init__(self, foc_client: FOCClient, cache_dir: str = "/models"):
        self._foc = foc_client
        self._cache_dir = Path(cache_dir)
        self._loaded_model: str | None = None

    async def ensure_model(self, model_id: str) -> str:
        """Ensure a model is available locally. Returns the local path."""
        # Check if already cached AND complete.
        cached_path = self._cache_dir / model_id.replace("/", "--")
        if is_model_complete(cached_path):
            logger.info("model found in cache", extra={"model": model_id})
            self._loaded_model = model_id
            return str(cached_path)
        if cached_path.exists():
            # Partial/corrupt cache from a crashed download — remove and re-fetch
            # rather than returning an unloadable directory.
            logger.warning("cached model dir incomplete, removing and re-downloading",
                           extra={"model": model_id, "path": str(cached_path)})
            shutil.rmtree(cached_path, ignore_errors=True)

        # Try to resolve and download via FOC
        if await self._foc.check_availability(model_id):
            manifest = await self._foc.resolve_model(model_id)
            path = await self._foc.download_weights(manifest, cached_path)
            self._loaded_model = model_id
            return str(path)

        # Fallback: assume model_id is a HuggingFace model path
        logger.info("model not in FOC, using as direct path",
                     extra={"model": model_id})
        self._loaded_model = model_id
        return model_id

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model
