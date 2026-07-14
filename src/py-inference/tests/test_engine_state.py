"""P3 regression: a pause arriving while the engine is STARTING (initial model
load) must not be dropped — the engine must end up PAUSED, not RUNNING.

Exercises the REAL VLLMEngine state machine by faking only the three methods
that touch vLLM/torch/nvidia-smi.
"""
import asyncio

import pytest

from src.inference.engine import VLLMEngine, EngineState, YieldUrgency


class FakeVLLMEngine(VLLMEngine):
    """VLLMEngine with hardware methods faked out for unit testing."""
    def __init__(self):
        super().__init__(device_ids=[0])
        self.create_calls = 0
        self.destroy_calls = 0
        self._gate = None  # optional asyncio.Event to stall _create_engine

    async def _create_engine(self):
        self.create_calls += 1
        self._engine = object()  # non-None sentinel
        if self._gate is not None:
            await self._gate.wait()

    async def _destroy_engine(self):
        self.destroy_calls += 1
        self._engine = None

    async def _wait_for_vram_release(self, timeout: float = 20.0):
        return  # no GPU in tests


@pytest.mark.asyncio
async def test_pause_during_starting_is_honored():
    eng = FakeVLLMEngine()
    eng._gate = asyncio.Event()  # hold _create_engine open to simulate slow load

    start_task = asyncio.create_task(eng.start("model-x"))

    # Wait until we're inside the STARTING load.
    for _ in range(100):
        if eng._state == EngineState.STARTING and eng.create_calls == 1:
            break
        await asyncio.sleep(0.001)
    assert eng._state == EngineState.STARTING

    # A yield arrives mid-load.
    await eng.pause(YieldUrgency.IMMEDIATE)
    assert eng._pending_pause == YieldUrgency.IMMEDIATE

    # Let the load finish.
    eng._gate.set()
    await start_task

    # The engine must have unloaded instead of coming up RUNNING.
    assert eng._state == EngineState.PAUSED, \
        "P3 regression: pause during STARTING was dropped, engine is RUNNING"
    assert eng.destroy_calls == 1
    assert eng._pending_pause is None


@pytest.mark.asyncio
async def test_normal_start_comes_up_running():
    eng = FakeVLLMEngine()
    await eng.start("model-x")
    assert eng._state == EngineState.RUNNING
    assert eng.destroy_calls == 0


@pytest.mark.asyncio
async def test_pause_resume_roundtrip():
    eng = FakeVLLMEngine()
    await eng.start("model-x")
    assert eng._state == EngineState.RUNNING

    await eng.pause(YieldUrgency.IMMEDIATE)
    assert eng._state == EngineState.PAUSED
    assert eng.destroy_calls == 1

    await eng.resume()
    assert eng._state == EngineState.RUNNING
    assert eng.create_calls == 2  # initial start + resume


@pytest.mark.asyncio
async def test_pause_during_loading_resume_path():
    """Existing behavior: pause during LOADING (resume) must still unload."""
    eng = FakeVLLMEngine()
    await eng.start("model-x")
    await eng.pause(YieldUrgency.IMMEDIATE)
    assert eng._state == EngineState.PAUSED

    eng._gate = asyncio.Event()
    resume_task = asyncio.create_task(eng.resume())
    for _ in range(100):
        if eng._state == EngineState.LOADING:
            break
        await asyncio.sleep(0.001)
    assert eng._state == EngineState.LOADING

    await eng.pause(YieldUrgency.IMMEDIATE)  # pause during the reload
    eng._gate.set()
    await resume_task
    assert eng._state == EngineState.PAUSED
