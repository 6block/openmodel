"""Listener coverage: yield-urgency selection (NORMAL vs IMMEDIATE), unknown-state
rejection, and fail-safe clearing on reconnect (guards 'paused forever')."""
import asyncio
import threading
import time

import pytest

from conftest import FakeSchedulerClient
from src.scheduler_client.listener import (
    SchedulerListener,
    _GPU_STATE_AVAILABLE,
    _GPU_STATE_YIELDING,
    _GPU_STATE_WINDOW_POST,
    _GPU_STATE_WINNING_POST,
)
from src.inference.engine import YieldUrgency


class UrgencyEngine:
    def __init__(self):
        self.urgencies = []

    def request_pause(self):
        pass

    async def resume(self):
        pass

    async def pause(self, urgency=None):
        self.urgencies.append(urgency)


@pytest.fixture
def running_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)


def _listener(loop, engine, client=None):
    return SchedulerListener(client or FakeSchedulerClient(), engine, main_loop=loop, poll_interval=0.05)


def test_yielding_uses_normal_urgency(running_loop):
    e = UrgencyEngine()
    assert _listener(running_loop, e)._apply_state(_GPU_STATE_YIELDING) is True
    assert e.urgencies == [YieldUrgency.NORMAL]  # graceful drain


def test_window_post_uses_immediate(running_loop):
    e = UrgencyEngine()
    assert _listener(running_loop, e)._apply_state(_GPU_STATE_WINDOW_POST) is True
    assert e.urgencies == [YieldUrgency.IMMEDIATE]


def test_winning_post_uses_immediate(running_loop):
    e = UrgencyEngine()
    assert _listener(running_loop, e)._apply_state(_GPU_STATE_WINNING_POST) is True
    assert e.urgencies == [YieldUrgency.IMMEDIATE]


def test_unknown_state_returns_false(running_loop):
    e = UrgencyEngine()
    assert _listener(running_loop, e)._apply_state(999) is False


def test_reconnect_clears_failsafe(running_loop):
    client = FakeSchedulerClient()
    e = UrgencyEngine()
    l = _listener(running_loop, e, client)
    l._failsafe_timeout = 0.0  # trigger the fail-safe on the first disconnected poll

    # Disconnect → fail-safe pauses inference.
    client.raise_on_get = True
    l._poll()
    assert l._failsafe_triggered is True
    assert e.urgencies == [YieldUrgency.IMMEDIATE]  # fail-safe yields immediately

    # Reconnect → fail-safe MUST clear (else inference stays paused forever).
    client.raise_on_get = False
    client.set_state(_GPU_STATE_AVAILABLE)
    l._poll()
    assert l._failsafe_triggered is False
