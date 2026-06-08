"""Background listener for scheduler events, bridging gRPC to engine lifecycle.

IMPORTANT: All engine operations (pause/resume) are scheduled on the main
asyncio event loop via run_coroutine_threadsafe(). This ensures vLLM engine
creation/destruction always happens on the same event loop as generate(),
preventing cross-loop deadlocks.
"""

import asyncio
import logging
import threading
import traceback
import time

from ..inference.engine import InferenceEngine, YieldUrgency
from .grpc_client import SchedulerClient

logger = logging.getLogger(__name__)

# Map protobuf GpuState values to actions
_GPU_STATE_AVAILABLE = 1
_GPU_STATE_YIELDING = 2
_GPU_STATE_WINDOW_POST = 3
_GPU_STATE_WINNING_POST = 4

# Map protobuf YieldUrgency values
_URGENCY_NORMAL = 0
_URGENCY_IMMEDIATE = 1

_YIELD_STATES = (_GPU_STATE_YIELDING, _GPU_STATE_WINDOW_POST, _GPU_STATE_WINNING_POST)


def _proto_urgency_to_engine(urgency: int) -> YieldUrgency:
    return {
        _URGENCY_NORMAL: YieldUrgency.NORMAL,
        _URGENCY_IMMEDIATE: YieldUrgency.IMMEDIATE,
    }.get(urgency, YieldUrgency.NORMAL)


_STATE_NAMES = {
    0: "UNKNOWN",
    _GPU_STATE_AVAILABLE: "AVAILABLE",
    _GPU_STATE_YIELDING: "YIELDING",
    _GPU_STATE_WINDOW_POST: "WINDOW_POST",
    _GPU_STATE_WINNING_POST: "WINNING_POST",
}


class SchedulerListener:
    """Listens to scheduler gRPC events and controls the inference engine.

    Uses a polling approach (GetGpuSchedule) for reliability,
    with streaming (SubscribeScheduleEvents) as an optional enhancement.

    Engine operations are dispatched to the main event loop to ensure
    vLLM engine lifecycle stays on a single loop (avoids cross-loop hangs).
    """

    def __init__(self, client: SchedulerClient, engine: InferenceEngine,
                 main_loop: asyncio.AbstractEventLoop,
                 poll_interval: float = 3.0):
        self._client = client
        self._engine = engine
        self._main_loop = main_loop
        self._running = False
        self._thread = None
        self._poll_interval = poll_interval
        self._last_state: int | None = None
        self._connected = True
        self._disconnect_since: float | None = None  # monotonic time of first failure
        self._failsafe_timeout = 60.0  # pause inference after 60s of disconnect
        self._failsafe_triggered = False

    def start(self):
        """Start the listener in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("scheduler listener started (polling mode, interval=%.1fs)",
                     self._poll_interval)

    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        """Background thread that polls the scheduler for GPU state."""
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logger.error("scheduler poll error: %s\n%s", e, traceback.format_exc())

            # Sleep in small increments so we can stop quickly
            for _ in range(int(self._poll_interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def _poll(self):
        """Poll the scheduler for current GPU state and react to changes."""
        try:
            schedule = self._client.get_gpu_schedule()
        except Exception as e:
            logger.error("failed to get GPU schedule: %s", e)
            self._connected = False

            # Track disconnect duration for fail-safe
            now = time.monotonic()
            if self._disconnect_since is None:
                self._disconnect_since = now

            elapsed = now - self._disconnect_since
            if not self._failsafe_triggered and elapsed >= self._failsafe_timeout:
                logger.warning(
                    "scheduler disconnected for %.0fs — fail-safe: pausing inference to protect mining",
                    elapsed)
                self._failsafe_triggered = True
                self._last_state = _GPU_STATE_WINDOW_POST
                self._apply_state(_GPU_STATE_WINDOW_POST)
            return

        # Connection restored
        if not self._connected:
            self._connected = True
            disconnect_duration = 0.0
            if self._disconnect_since is not None:
                disconnect_duration = time.monotonic() - self._disconnect_since
            self._disconnect_since = None
            logger.info("reconnected to scheduler (was disconnected %.1fs)", disconnect_duration)

            if self._failsafe_triggered:
                self._failsafe_triggered = False
                # Force re-apply whatever state the scheduler returns,
                # even if it matches _last_state, to ensure resume happens.
                self._last_state = None
                logger.info("fail-safe cleared — will apply scheduler state immediately")

        state = schedule.state
        state_name = _STATE_NAMES.get(state, f"UNKNOWN({state})")

        if state == self._last_state:
            return  # No change

        old_name = _STATE_NAMES.get(self._last_state, "NONE")
        logger.info("GPU state changed: %s -> %s (msg: %s)",
                     old_name, state_name, schedule.message)
        self._last_state = state

        self._apply_state(state)

    def _apply_state(self, state: int):
        """Apply a GPU state change to the inference engine.

        Operations are dispatched to the main event loop via
        run_coroutine_threadsafe() to ensure vLLM engine lifecycle
        stays on the same loop as generate() requests.

        Note: pause() destroys the vLLM engine and releases GPU VRAM.
        resume() reloads the model, which may take 10-30 seconds.
        future.result() blocks this thread until completion, which is
        fine since no other state changes matter during unload/reload.
        """
        try:
            if state == _GPU_STATE_AVAILABLE:
                logger.info("resuming inference — this may take 10-30s for model reload...")
                t0 = time.monotonic()
                future = asyncio.run_coroutine_threadsafe(
                    self._engine.resume(), self._main_loop
                )
                future.result(timeout=120)  # Block until resume completes
                elapsed = time.monotonic() - t0
                logger.info("inference RESUMED (took %.1fs)", elapsed)

            elif state in _YIELD_STATES:
                # Use IMMEDIATE urgency for WINDOW_POST and WINNING_POST,
                # NORMAL for YIELDING
                if state == _GPU_STATE_YIELDING:
                    urgency = YieldUrgency.NORMAL
                else:
                    urgency = YieldUrgency.IMMEDIATE

                logger.info("pausing inference — will unload model and release GPU VRAM...")
                t0 = time.monotonic()
                future = asyncio.run_coroutine_threadsafe(
                    self._engine.pause(urgency), self._main_loop
                )
                future.result(timeout=120)  # Block until pause completes
                elapsed = time.monotonic() - t0
                logger.info("inference PAUSED (state=%s, urgency=%s, took %.1fs)",
                            _STATE_NAMES.get(state), urgency.name, elapsed)
            else:
                logger.warning("unknown GPU state: %d", state)
        except Exception as e:
            logger.error("failed to apply state %d: %s\n%s",
                         state, e, traceback.format_exc())
