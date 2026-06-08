"""Model lifecycle manager."""

import logging
from pathlib import Path

from .foc_client import FOCClient

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and caching."""

    def __init__(self, foc_client: FOCClient, cache_dir: str = "/models"):
        self._foc = foc_client
        self._cache_dir = Path(cache_dir)
        self._loaded_model: str | None = None

    async def ensure_model(self, model_id: str) -> str:
        """Ensure a model is available locally. Returns the local path."""
        # Check if already cached
        cached_path = self._cache_dir / model_id.replace("/", "--")
        if cached_path.exists():
            logger.info("model found in cache", extra={"model": model_id})
            self._loaded_model = model_id
            return str(cached_path)

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
