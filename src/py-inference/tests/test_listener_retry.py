"""P1 regression tests: a failed pause/resume must be retried on the next poll,
and _last_state must only advance after a successful transition.
"""
import asyncio
import threading
import time

import pytest

from conftest import FakeSchedulerClient
from src.scheduler_client.listener import (
    SchedulerListener,
    _GPU_STATE_AVAILABLE,
    _GPU_STATE_WINDOW_POST,
)


class FlakyEngine:
    """Engine whose pause() fails the first `fail_times` calls, then succeeds."""
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.pause_calls = 0
        self.resume_calls = 0
        self.paused = False

    async def pause(self, urgency=None):
        self.pause_calls += 1
        if self.pause_calls <= self.fail_times:
            raise RuntimeError("simulated pause failure")
        self.paused = True

    async def resume(self):
        self.resume_calls += 1
        self.paused = False


@pytest.fixture
def running_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)


def _make_listener(client, engine, loop):
    return SchedulerListener(client, engine, main_loop=loop, poll_interval=0.05)


def test_failed_pause_is_retried(running_loop):
    """A pause that throws must NOT advance _last_state, so the next poll retries."""
    client = FakeSchedulerClient()
    engine = FlakyEngine(fail_times=1)  # first pause fails, second succeeds
    listener = _make_listener(client, engine, running_loop)

    client.set_state(_GPU_STATE_WINDOW_POST)

    # First poll: pause fails → _last_state must stay unchanged (not WINDOW_POST)
    listener._poll()
    assert engine.pause_calls == 1
    assert engine.paused is False
    assert listener._last_state != _GPU_STATE_WINDOW_POST, \
        "P1 regression: _last_state advanced despite failed pause"

    # Second poll: same state still pending → retry → succeeds
    listener._poll()
    assert engine.pause_calls == 2
    assert engine.paused is True
    assert listener._last_state == _GPU_STATE_WINDOW_POST


def test_successful_pause_advances_state(running_loop):
    client = FakeSchedulerClient()
    engine = FlakyEngine(fail_times=0)
    listener = _make_listener(client, engine, running_loop)

    client.set_state(_GPU_STATE_WINDOW_POST)
    listener._poll()
    assert engine.pause_calls == 1
    assert listener._last_state == _GPU_STATE_WINDOW_POST

    # No change on a repeat poll
    listener._poll()
    assert engine.pause_calls == 1  # not called again


def test_failsafe_pause_retries_on_failure(running_loop):
    """When the scheduler is unreachable, a failed fail-safe pause must retry."""
    client = FakeSchedulerClient()
    client.raise_on_get = True
    engine = FlakyEngine(fail_times=1)
    listener = _make_listener(client, engine, running_loop)
    listener._failsafe_timeout = 0.0  # trigger immediately on first disconnect

    # First poll: disconnect → fail-safe pause attempted but fails
    listener._poll()
    assert engine.pause_calls == 1
    assert listener._failsafe_triggered is False, \
        "P1 regression: fail-safe marked triggered despite failed pause"

    # Second poll: still disconnected → retry → succeeds
    listener._poll()
    assert engine.pause_calls == 2
    assert engine.paused is True
    assert listener._failsafe_triggered is True
