"""P5 regression: the listener must report engine status to the scheduler on
each poll, so a stuck pause (is_running stays True during a yield) is visible."""
import asyncio
import threading

import pytest

from conftest import FakeSchedulerClient
from src.scheduler_client.listener import SchedulerListener, _GPU_STATE_WINDOW_POST
from src.inference.engine import EngineState, EngineStatus, YieldUrgency


class StatusEngine:
    def __init__(self, state=EngineState.RUNNING):
        self._state = state
        self.paused = False

    async def pause(self, urgency=YieldUrgency.NORMAL):
        self._state = EngineState.PAUSED
        self.paused = True

    async def resume(self):
        self._state = EngineState.RUNNING

    def request_pause(self):
        pass

    def status(self):
        return EngineStatus(state=self._state, active_requests=2,
                            loaded_model="Qwen/test", gpu_id=0)


@pytest.fixture
def running_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)


def test_status_reported_each_poll(running_loop):
    client = FakeSchedulerClient()
    engine = StatusEngine(EngineState.RUNNING)
    listener = SchedulerListener(client, engine, main_loop=running_loop, poll_interval=0.05)

    listener._poll()
    assert len(client.reported) == 1
    rep = client.reported[0]
    assert rep["is_running"] is True
    assert rep["active_requests"] == 2
    assert rep["model"] == "Qwen/test"


def test_report_reflects_paused_state(running_loop):
    client = FakeSchedulerClient()
    engine = StatusEngine(EngineState.RUNNING)
    listener = SchedulerListener(client, engine, main_loop=running_loop, poll_interval=0.05)

    # Pause via a yield state, then poll again — report must show not running.
    client.set_state(_GPU_STATE_WINDOW_POST)
    listener._poll()  # applies pause
    assert engine.paused is True

    client.reported.clear()
    listener._poll()  # next heartbeat
    assert client.reported[-1]["is_running"] is False


def test_report_failure_does_not_break_poll(running_loop):
    class BoomClient(FakeSchedulerClient):
        def report_status(self, **kwargs):
            raise RuntimeError("report boom")

    client = BoomClient()
    engine = StatusEngine(EngineState.RUNNING)
    listener = SchedulerListener(client, engine, main_loop=running_loop, poll_interval=0.05)
    # Should not raise despite report_status failing.
    listener._poll()
