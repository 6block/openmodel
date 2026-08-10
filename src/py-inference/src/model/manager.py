"""Model lifecycle manager."""

import asyncio
import logging
import os
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
        # One lock per model id: a background prefetch and a request-triggered
        # switch may target the same weights; without this they would race the
        # same cache directory mid-download.
        self._dl_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        lock = self._dl_locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._dl_locks[model_id] = lock
        return lock

    def _cache_path_for(self, model_id: str) -> Path:
        """Canonical on-disk directory for a model. Path-form ids keep their
        basename ("/models/Org--Name" → cache/Org--Name); only HF ids get the
        "/"→"--" mapping. Blindly mangling a path-form id produced a
        "--models--Org--Name" directory (found live): consistent on both sides
        so it worked, but it doubles disk when the same model is later fetched
        under its HF spelling, and matches no operator convention."""
        if model_id.startswith("/"):
            return self._cache_dir / Path(model_id).name
        return self._cache_dir / model_id.replace("/", "--")

    def _present_path(self, model_id: str) -> Path | None:
        """Local path of a complete copy of model_id, or None. Mirrors the
        resolution order of ensure_model without any side effects."""
        if model_id.startswith("/"):
            direct = Path(model_id)
            for candidate in (direct, self._cache_dir / direct.name):
                if is_model_complete(candidate):
                    return candidate
        cached = self._cache_path_for(model_id)
        if is_model_complete(cached):
            return cached
        return None

    async def ensure_model(self, model_id: str) -> str:
        """Ensure a model is available locally. Returns the local path."""
        # A request may name the model by its LOCAL PATH — the whole fleet
        # advertises loaded models that way ("/models/Qwen--Qwen2.5-3B-Instruct"),
        # so that is exactly what a gateway-triggered switch sends. Mangling a
        # path through the HF-id replace() below yields "--models--Qwen--…", a
        # directory that can never exist, so every switch toward a path-named
        # model skipped the local weights and died trying to download instead.
        cached_path = self._cache_path_for(model_id)
        # Fast path — but ONLY while no download of this model is in flight. A
        # background prefetch may be extracting into the cache dir right now: a
        # half-extracted directory can pass is_model_complete (config.json plus
        # a partially written weight file) and must be neither served nor
        # rmtree'd mid-write. If the lock is held, fall through and queue on it;
        # the post-lock re-check takes the finished copy.
        if not self._lock_for(model_id).locked():
            if model_id.startswith("/"):
                direct = Path(model_id)
                for candidate in (direct, self._cache_dir / direct.name):
                    if is_model_complete(candidate):
                        logger.info("model found by path", extra={"model": model_id})
                        self._loaded_model = model_id
                        return str(candidate)
            # Check if already cached AND complete.
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

        # Try to resolve and download via FOC. Under the per-model lock: a
        # background prefetch may already be fetching these exact weights, in
        # which case we wait and take the cache hit instead of downloading twice.
        async with self._lock_for(model_id):
            present = self._present_path(model_id)
            if present is not None:
                logger.info("model appeared while waiting for download lock",
                            extra={"model": model_id})
                self._loaded_model = model_id
                return str(present)
            try:
                if await self._foc.check_availability(model_id):
                    manifest = await self._foc.resolve_model(model_id)
                    path = await self._foc.download_weights(manifest, cached_path)
                    self._loaded_model = model_id
                    return str(path)
            except Exception as e:  # noqa: BLE001 — a dead provider must not be terminal
                logger.warning("FOC fetch failed for %s: %s", model_id, e)

            # FOC has nothing (or its single retrieval provider is down — it
            # happened, and it bricked every first boot). Path-form ids can
            # fall back to Hugging Face via the canonical basename transform;
            # the copy is fetched into the SAME canonical dir, but is NOT
            # covered by any FOC-pinned digest — say so loudly.
            hf_id = self._derived_hf_id(model_id)
            if hf_id and self._hf_fallback_enabled():
                logger.warning(
                    "falling back to Hugging Face for %s (as %s) — this copy is "
                    "not covered by the FOC-pinned digest", model_id, hf_id)
                try:
                    await asyncio.to_thread(self._hf_fetch_dir, hf_id, cached_path)
                except Exception as e:  # noqa: BLE001 — gated repo / no egress
                    logger.warning("HF fallback failed for %s: %s", model_id, e)
                if is_model_complete(cached_path):
                    self._loaded_model = model_id
                    return str(cached_path)

        # Fallback: assume model_id is a HuggingFace model path
        logger.info("model not in FOC, using as direct path",
                     extra={"model": model_id})
        self._loaded_model = model_id
        return model_id

    async def prefetch(self, model_id: str) -> str:
        """Ensure the weights for a claimed model are on local disk WITHOUT
        loading or switching anything.

        Serving-path downloads happen inside a request-triggered switch, where
        the triggering request (an admission probe or a paying user) times out
        long before a multi-minute download finishes. Prefetching at startup
        moves the download out of the request path entirely: a later switch
        only pays the load time.

        Returns one of: "present", "foc-downloaded", "hf-ensured",
        "hf-downloaded" (HF fallback into the canonical dir), "unavailable".
        """
        if self._present_path(model_id) is not None:
            return "present"
        async with self._lock_for(model_id):
            if self._present_path(model_id) is not None:
                return "present"  # a concurrent switch fetched it while we waited

            cached_path = self._cache_path_for(model_id)
            if cached_path.exists():
                # Same guard as ensure_model: a half-written directory from a
                # crashed download must not shadow a fresh fetch.
                logger.warning("prefetch: incomplete cached dir, removing",
                               extra={"model": model_id, "path": str(cached_path)})
                shutil.rmtree(cached_path, ignore_errors=True)

            try:
                if await self._foc.check_availability(model_id):
                    manifest = await self._foc.resolve_model(model_id)
                    await self._foc.download_weights(manifest, cached_path)
                    return "foc-downloaded"
            except Exception as e:  # noqa: BLE001 — a dead provider must not be terminal
                logger.warning("FOC prefetch failed for %s: %s", model_id, e)

            if "/" in model_id and not model_id.startswith("/"):
                # HF-form id: pull the snapshot into the HF cache so the later
                # engine load finds every file locally. Honors HF_ENDPOINT /
                # HF_TOKEN / HF_HOME from the environment, exactly like the
                # engine's own downloader. Blocking call → worker thread.
                await asyncio.to_thread(self._hf_snapshot, model_id)
                return "hf-ensured"

            # Path-form id: fall back to Hugging Face via the canonical
            # basename transform ("/models/Org--Name" → "Org/Name") when FOC
            # has nothing or its provider is down. Fetched into the canonical
            # dir; NOT covered by any FOC-pinned digest.
            hf_id = self._derived_hf_id(model_id)
            if hf_id and self._hf_fallback_enabled():
                logger.warning(
                    "prefetch falling back to Hugging Face for %s (as %s) — this "
                    "copy is not covered by the FOC-pinned digest", model_id, hf_id)
                await asyncio.to_thread(self._hf_fetch_dir, hf_id, cached_path)
                return "hf-downloaded"

            # Nothing to fetch from (underivable path, or fallback disabled).
            # The admission probe for this claim stays pending until the
            # operator supplies the weights.
            return "unavailable"

    @staticmethod
    def _hf_snapshot(model_id: str) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_id)

    @staticmethod
    def _derived_hf_id(model_id: str) -> str | None:
        """Org/Name derived from a path-form id's basename ("/models/Org--Name"),
        or None when no such derivation exists. Same transform the gateway uses
        to canonicalize model ids."""
        if not model_id.startswith("/"):
            return None
        base = Path(model_id).name
        if "--" in base:
            return base.replace("--", "/", 1)
        return None

    @staticmethod
    def _hf_fallback_enabled() -> bool:
        # Operators without HF egress can disable the fallback outright.
        return os.environ.get("HF_FALLBACK", "1") != "0"

    @staticmethod
    def _hf_fetch_dir(hf_id: str, dest: Path) -> None:
        """Materialize an HF snapshot INTO the canonical cache dir (real files,
        not hub-cache symlinks) so path-form lookups and operator conventions
        keep working."""
        from huggingface_hub import snapshot_download

        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=hf_id, local_dir=str(dest))

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model
