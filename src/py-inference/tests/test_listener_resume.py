"""P2 (listener side): resume is dispatched non-blocking so a yield arriving
mid-resume preempts it via request_pause(); a failed background resume retries."""
import asyncio
import threading
import time

import pytest

from conftest import FakeSchedulerClient
from src.scheduler_client.listener import (
    SchedulerListener,
    _GPU_STATE_AVAILABLE,
    _GPU_STATE_WINNING_POST,
)


class PoolLikeEngine:
    def __init__(self):
        self._pause_requested = False
        self.resume_count = 0
        self.resume_aborted = False
        self.paused = False
        self.resume_should_fail = False
        self.resume_delay = 0.3

    def request_pause(self):
        self._pause_requested = True

    async def resume(self):
        self.resume_count += 1
        self._pause_requested = False
        if self.resume_should_fail:
            raise RuntimeError("simulated resume failure")
        await asyncio.sleep(self.resume_delay)
        if self._pause_requested:
            self.resume_aborted = True

    async def pause(self, urgency=None):
        self.paused = True


@pytest.fixture
def running_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)


def test_yield_during_resume_preempts(running_loop):
    client = FakeSchedulerClient()
    engine = PoolLikeEngine()
    listener = SchedulerListener(client, engine, main_loop=running_loop, poll_interval=0.05)

    # Resume dispatched non-blocking.
    client.set_state(_GPU_STATE_AVAILABLE)
    listener._poll()
    resume_future = listener._resume_future
    assert resume_future is not None
    time.sleep(0.05)
    assert engine.resume_count == 1

    # Yield arrives while resume is still running → must preempt + pause.
    client.set_state(_GPU_STATE_WINNING_POST)
    listener._poll()
    assert engine.paused is True

    # The in-flight resume should observe the pause request and abort.
    resume_future.result(timeout=2)
    assert engine.resume_aborted is True, \
        "P2 regression: resume not preempted by mid-resume yield"


def test_failed_background_resume_retries(running_loop):
    client = FakeSchedulerClient()
    engine = PoolLikeEngine()
    engine.resume_delay = 0.0
    engine.resume_should_fail = True
    listener = SchedulerListener(client, engine, main_loop=running_loop, poll_interval=0.05)

    client.set_state(_GPU_STATE_AVAILABLE)
    listener._poll()  # dispatch resume (will fail)
    time.sleep(0.1)  # let it fail
    assert engine.resume_count == 1

    # Next poll: still AVAILABLE; the failed future should force a re-dispatch.
    engine.resume_should_fail = False
    listener._poll()
    time.sleep(0.1)
    assert engine.resume_count == 2, "failed background resume was not retried"
