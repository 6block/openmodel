"""Mock FOC client for development without real Filecoin Orbit Chain."""

import asyncio
import logging
from pathlib import Path

from .foc_client import FOCClient, ModelManifest

logger = logging.getLogger(__name__)


class MockFOCClient(FOCClient):
    """Mock implementation that reads from local model paths."""

    def __init__(self, model_registry: dict[str, str],
                 simulate_delay: bool = False):
        """
        Args:
            model_registry: Mapping of model_id -> local filesystem path.
            simulate_delay: If True, simulate network download delay.
        """
        self._registry: dict[str, Path] = {
            k: Path(v) for k, v in model_registry.items()
        }
        self._simulate_delay = simulate_delay

    async def resolve_model(self, model_id: str) -> ModelManifest:
        if model_id not in self._registry:
            raise ValueError(f"Model '{model_id}' not found in mock registry. "
                             f"Available: {list(self._registry.keys())}")

        path = self._registry[model_id]
        return ModelManifest(
            model_id=model_id,
            size_bytes=0,  # Unknown for local files
            checksum_sha256="mock-checksum",
            source_uri=f"file://{path}",
            quantization=None,
        )

    async def download_weights(self, manifest: ModelManifest, dest: Path) -> Path:
        source = manifest.source_uri.replace("file://", "")
        source_path = Path(source)

        if self._simulate_delay and manifest.size_bytes > 0:
            # Simulate ~1 GB/s download speed
            delay = manifest.size_bytes / (1024 ** 3)
            logger.info("simulating download delay",
                        extra={"seconds": delay, "model": manifest.model_id})
            await asyncio.sleep(delay)

        # For mock, just return the source path directly (no actual copy)
        logger.info("mock download complete",
                    extra={"model": manifest.model_id, "path": str(source_path)})
        return source_path

    async def check_availability(self, model_id: str) -> bool:
        return model_id in self._registry


def create_foc_client(config: dict) -> FOCClient:
    """Factory function to create the appropriate FOC client."""
    backend = config.get("backend", "mock")

    if backend == "mock":
        mock_config = config.get("mock", {})
        return MockFOCClient(
            model_registry=mock_config.get("model_registry", {}),
            simulate_delay=mock_config.get("simulate_delay", False),
        )
    elif backend == "real":
        from .foc_real import RealFOCClient
        real_config = config.get("real", {})
        return RealFOCClient(
            bridge_url=real_config.get("bridge_url", "http://127.0.0.1:3100"),
            piece_cid_registry=real_config.get("piece_cid_registry", {}),
        )
    else:
        raise ValueError(f"Unknown FOC backend: {backend}")
