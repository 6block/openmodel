"""P4 regression: _wait_for_vram_release must not proceed immediately on a
single transient nvidia-smi failure, but must still proceed if nvidia-smi is
genuinely unavailable (dev/mock) or after the timeout."""
import pytest

from src.inference.engine import VLLMEngine


class VramEngine(VLLMEngine):
    """Engine with a scripted _query_gpu_free_ratio sequence."""
    def __init__(self, ratios):
        super().__init__(device_ids=[0], gpu_memory_utilization=0.85)
        self._ratios = list(ratios)
        self.queries = 0

    def _query_gpu_free_ratio(self, gpu_id):
        self.queries += 1
        if self._ratios:
            return self._ratios.pop(0)
        return self._ratios_default if hasattr(self, "_ratios_default") else None


@pytest.mark.asyncio
async def test_transient_failure_then_free():
    # First query fails (transient), then VRAM is free → must wait, not bail.
    eng = VramEngine([None, 0.9])
    await eng._wait_for_vram_release(timeout=5.0)
    assert eng.queries >= 2, "P4 regression: bailed on first transient nvidia-smi failure"


@pytest.mark.asyncio
async def test_persistent_unavailable_proceeds():
    # nvidia-smi never works → after a few retries, proceed (dev/mock behavior).
    eng = VramEngine([None, None, None, None, None])
    eng._ratios_default = None
    await eng._wait_for_vram_release(timeout=5.0)
    # Should give up after ~3 consecutive failures, not poll the full window.
    assert eng.queries <= 4


@pytest.mark.asyncio
async def test_waits_until_vram_free():
    # VRAM not free for a couple polls, then frees.
    eng = VramEngine([0.1, 0.2, 0.9])
    await eng._wait_for_vram_release(timeout=5.0)
    assert eng.queries == 3


@pytest.mark.asyncio
async def test_timeout_proceeds_when_never_free():
    # VRAM stays occupied → proceed after timeout (cannot hang forever).
    eng = VramEngine([])
    eng._ratios_default = 0.1  # always 10% free, never enough
    await eng._wait_for_vram_release(timeout=1.0)
    assert eng.queries >= 1
