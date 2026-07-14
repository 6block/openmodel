"""Unit tests for VLLMEngine.switch_model — the happy path and the already-loaded
no-op. (The rollback branch is superseded by _do_reload's os._exit on persistent
reload failure, so it is not unit-tested here — see m1-m2-coverage doc.)"""
import asyncio

from src.inference.engine import VLLMEngine, EngineState


class SwitchFake(VLLMEngine):
    """VLLMEngine with the three hardware methods faked (all succeed)."""
    def __init__(self):
        super().__init__(device_ids=[0])

    async def _create_engine(self):
        self._engine = object()

    async def _destroy_engine(self):
        self._engine = None

    async def _wait_for_vram_release(self, timeout: float = 20.0):
        return


def test_switch_model_happy():
    e = SwitchFake()
    asyncio.run(e.start("old-model"))
    assert e.current_model == "old-model"
    asyncio.run(e.switch_model("new-model"))
    assert e.current_model == "new-model"
    assert e._state == EngineState.RUNNING


def test_switch_model_noop_when_already_loaded():
    e = SwitchFake()
    asyncio.run(e.start("m"))
    asyncio.run(e.switch_model("m"))  # same model + RUNNING → early return
    assert e.current_model == "m"
    assert e._state == EngineState.RUNNING
