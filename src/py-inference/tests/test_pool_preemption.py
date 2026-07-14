"""P2 regression: when a pause arrives during a sequential multi_instance
resume, the resume loop must abort so not-yet-resumed engines are NOT loaded
onto a GPU that mining now needs."""
import asyncio

import pytest

from src.inference.engine import EngineState, YieldUrgency
from src.inference.engine_pool import EnginePool


class SlowEngine:
    """Minimal engine modeling the real VLLMEngine's _pending_pause behavior:
    a pause during LOADING is deferred and applied when the load finishes."""
    def __init__(self, gpu_id, resume_delay=0.2):
        self.gpu_id = gpu_id
        self._state = EngineState.PAUSED
        self.resume_delay = resume_delay
        self.resume_started = 0
        self._pending_pause = False

    async def start(self, model_path):
        self._state = EngineState.RUNNING

    async def pause(self, urgency=YieldUrgency.NORMAL):
        if self._state == EngineState.LOADING:
            self._pending_pause = True  # deferred, applied after load (real engine)
            return
        self._state = EngineState.PAUSED

    async def resume(self):
        self.resume_started += 1
        self._pending_pause = False
        self._state = EngineState.LOADING
        await asyncio.sleep(self.resume_delay)
        if self._pending_pause:
            self._state = EngineState.PAUSED
            self._pending_pause = False
        else:
            self._state = EngineState.RUNNING

    def is_available(self):
        return self._state == EngineState.RUNNING

    def status(self):
        from src.inference.engine import EngineStatus
        return EngineStatus(state=self._state, active_requests=0,
                            loaded_model="m", gpu_id=self.gpu_id)


@pytest.mark.asyncio
async def test_pause_during_multi_instance_resume_aborts_loop():
    engines = [SlowEngine(i, resume_delay=0.2) for i in range(8)]
    pool = EnginePool(mode="multi_instance", engines=engines)

    resume_task = asyncio.create_task(pool.resume())

    # Let the first couple of engines start resuming, then request a pause.
    await asyncio.sleep(0.3)
    pool.request_pause()
    await pool.pause(YieldUrgency.IMMEDIATE)

    await resume_task  # resume loop should have aborted

    # Not all engines should have been resumed (loop aborted early).
    started = sum(e.resume_started for e in engines)
    assert started < 8, f"P2 regression: resume loaded all {started} engines despite pause"

    # After pause, ALL engines must be paused (none left running on the GPU).
    assert all(e._state == EngineState.PAUSED for e in engines), \
        "P2 regression: an engine is still RUNNING after pause during resume"


@pytest.mark.asyncio
async def test_clean_resume_loads_all_when_no_pause():
    engines = [SlowEngine(i, resume_delay=0.01) for i in range(4)]
    pool = EnginePool(mode="multi_instance", engines=engines)
    await pool.resume()
    assert all(e._state == EngineState.RUNNING for e in engines)
    assert sum(e.resume_started for e in engines) == 4


@pytest.mark.asyncio
async def test_request_pause_sets_flag():
    engines = [SlowEngine(0)]
    pool = EnginePool(mode="multi_instance", engines=engines)
    assert pool._pause_requested is False
    pool.request_pause()
    assert pool._pause_requested is True
    # A subsequent resume clears it.
    await pool.resume()
    assert pool._pause_requested is False
