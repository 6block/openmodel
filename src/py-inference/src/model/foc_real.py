"""Real FOC client that retrieves models from Filecoin Onchain Cloud via the foc-bridge."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from .foc_client import FOCClient, ModelManifest

logger = logging.getLogger(__name__)

# 30 minutes timeout for large model downloads on testnet
DOWNLOAD_TIMEOUT = 1800.0


class RealFOCClient(FOCClient):
    """FOC client that communicates with the foc-bridge (Node.js) service
    to upload/download models from Filecoin Onchain Cloud.

    The bridge wraps the Synapse SDK and exposes HTTP endpoints:
      POST /resolve          - Check if a PieceCID exists
      POST /download-model   - Download multi-part model as reassembled tar.gz
    """

    def __init__(self, bridge_url: str, piece_cid_registry: dict[str, list[str] | str]):
        """
        Args:
            bridge_url: URL of the foc-bridge service (e.g., "http://127.0.0.1:3100")
            piece_cid_registry: Mapping of model_id -> PieceCID or list of PieceCIDs.
                e.g., {"Qwen/Qwen2.5-1.5B-Instruct": ["cid1", "cid2", "cid3"]}
        """
        self._bridge_url = bridge_url.rstrip('/')
        # Normalize registry: always store as list
        self._registry: dict[str, list[str]] = {}
        for model_id, cids in piece_cid_registry.items():
            if isinstance(cids, str):
                self._registry[model_id] = [cids]
            else:
                self._registry[model_id] = list(cids)

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(
            connect=10.0,
            read=DOWNLOAD_TIMEOUT,
            write=30.0,
            pool=10.0,
        ))
        logger.info("RealFOCClient initialized",
                     extra={"bridge_url": bridge_url,
                            "registered_models": list(self._registry.keys())})

    def _get_cids(self, model_id: str) -> list[str] | None:
        """Look up the PieceCIDs for a model_id from the registry."""
        return self._registry.get(model_id)

    async def check_availability(self, model_id: str) -> bool:
        """Check if a model is available on FOC by resolving its first PieceCID."""
        cids = self._get_cids(model_id)
        if not cids:
            logger.debug("model %s not in piece_cid_registry", model_id)
            return False

        try:
            # Check first part as availability proxy
            resp = await self._client.post(
                f"{self._bridge_url}/resolve",
                json={"pieceCid": cids[0]},
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info("FOC model available: %s (%d parts, first_cid=%s)",
                            model_id, len(cids), cids[0][:20] + "...")
                return data.get("available", False)
            elif resp.status_code == 404:
                logger.warning("FOC model not found: %s (cid=%s)", model_id, cids[0])
                return False
            else:
                logger.warning("FOC resolve unexpected status %d for %s",
                               resp.status_code, model_id)
                return False
        except httpx.ConnectError:
            logger.error("cannot connect to foc-bridge at %s", self._bridge_url)
            return False
        except Exception as e:
            logger.error("FOC check_availability error: %s", e)
            return False

    async def resolve_model(self, model_id: str) -> ModelManifest:
        """Resolve a model ID to a downloadable manifest via FOC."""
        cids = self._get_cids(model_id)
        if not cids:
            raise ValueError(
                f"Model '{model_id}' not in piece_cid_registry. "
                f"Available: {list(self._registry.keys())}"
            )

        return ModelManifest(
            model_id=model_id,
            size_bytes=0,  # Will be known after download
            checksum_sha256="",  # FOC uses PieceCID for integrity
            source_uri=",".join(cids),  # All CIDs joined
            quantization=None,
        )

    async def download_weights(self, manifest: ModelManifest, dest: Path) -> Path:
        """Download model weights from FOC and extract to dest directory.

        Flow:
          1. POST /download-model with PieceCIDs and destPath
          2. Bridge downloads all parts from SP, writes to destPath on disk
          3. Returns JSON with path and size
          4. Extract tar.gz to dest directory
        """
        cids = manifest.source_uri.split(",")
        logger.info("downloading model from FOC: %s (%d parts)", manifest.model_id, len(cids))

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Use the shared models volume for temp download, so both
        # foc-bridge and inference containers can access the file.
        tmp_dir = dest.parent / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = str(tmp_dir / f"foc-model-{manifest.model_id.replace('/', '-')}.tar.gz")

        try:
            # Request multi-part download — bridge writes directly to disk
            resp = await self._client.post(
                f"{self._bridge_url}/download-model",
                json={"pieceCids": cids, "destPath": tmp_path},
            )
            resp.raise_for_status()
            result = resp.json()

            if not result.get("success"):
                raise RuntimeError(f"Bridge download failed: {result.get('error')}")

            size_mb = result.get("sizeBytes", 0) / (1024 * 1024)
            elapsed = result.get("elapsedSec", 0)
            logger.info("download complete: %.1f MB in %.1fs (%d parts)",
                        size_mb, elapsed, result.get("parts", 0))

            # Extract tar.gz to dest
            dest.mkdir(parents=True, exist_ok=True)
            logger.info("extracting model to %s", dest)
            subprocess.run(
                ["tar", "-xzf", tmp_path, "-C", str(dest)],
                check=True,
                capture_output=True,
            )

            logger.info("model extracted successfully: %s -> %s", manifest.model_id, dest)
            return dest

        except httpx.HTTPStatusError as e:
            logger.error("FOC download failed (HTTP %d): %s", e.response.status_code, e)
            raise
        except Exception as e:
            logger.error("FOC download error: %s", e)
            raise
        finally:
            # Clean up tar.gz
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
