"""Abstract interface for FOC (Filecoin Orbit Chain) model retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelManifest:
    """Describes a model available on the FOC network."""
    model_id: str
    size_bytes: int
    checksum_sha256: str
    source_uri: str           # e.g., "foc://llama-3-8b" or "file:///models/llama-3"
    quantization: str | None  # e.g., "awq", "gptq", None


class FOCClient(ABC):
    """Abstract base class for FOC model retrieval."""

    @abstractmethod
    async def resolve_model(self, model_id: str) -> ModelManifest:
        """Resolve a model ID to a downloadable manifest."""
        ...

    @abstractmethod
    async def download_weights(self, manifest: ModelManifest, dest: Path) -> Path:
        """Download model weights to dest directory. Returns path to weights."""
        ...

    @abstractmethod
    async def check_availability(self, model_id: str) -> bool:
        """Check if a model is available on the network."""
        ...
