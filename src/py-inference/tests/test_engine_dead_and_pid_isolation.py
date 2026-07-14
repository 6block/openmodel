"""Regression tests for the two soak findings around model switching:

1. _destroy_engine must only SIGKILL ITS OWN core subprocesses — the old code
   scanned multiprocessing.active_children() (= every sibling engine's core in
   multi_instance mode) and killed healthy engines during a per-engine switch.
2. A crashed engine core (vLLM EngineDeadError) must mark the engine DEAD,
   surface a retryable RuntimeError (→ API 503), be excluded from pool routing,
   and schedule automatic recovery — instead of staying "running" and turning
   every routed request into an unhandled 500.
3. switch_model must drain in-flight requests (while refusing new ones) before
   tearing the engine down.
"""
import asyncio
import sys
import types

import pytest

from src.inference.engine import VLLMEngine, MockEngine, EngineState
from src.inference.engine_pool import EnginePool


# generate() does `from vllm import SamplingParams` — stub it for GPU-less tests.
_vllm_stub = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
if not hasattr(_vllm_stub, "SamplingParams"):
    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    _vllm_stub.SamplingParams = _SamplingParams


class EngineDeadError(Exception):
    """Stand-in matching vLLM's class by NAME (detection is name-based)."""


class _FakeChild:
    def __init__(self, pid):
        self.pid = pid


class HardwareFake(VLLMEngine):
    """VLLMEngine with the vLLM-touching inner methods faked."""
    def __init__(self):
        super().__init__(device_ids=[0])

    async def _create_engine_inner(self):
        self._engine = object()

    async def _wait_for_vram_release(self, timeout: float = 20.0):
        return


# --- 1) destroy kills only own children -------------------------------------

def test_destroy_kills_only_own_child_pids(monkeypatch):
    async def run():
        e = HardwareFake()
        e._engine = types.SimpleNamespace(shutdown=lambda: None)
        e._child_pids = {11111}

        # active_children must NOT be consulted at destroy time anymore.
        scans = {"n": 0}
        def fake_children():
            scans["n"] += 1
            return [_FakeChild(11111), _FakeChild(22222), _FakeChild(33333)]
        import multiprocessing
        monkeypatch.setattr(multiprocessing, "active_children", fake_children)

        # Make the 5s/1s destroy waits instant.
        real_sleep = asyncio.sleep
        monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: real_sleep(0))

        kills = []
        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if sig == 0:
                return  # "still alive" → triggers SIGKILL path
        import os as _os
        monkeypatch.setattr(_os, "kill", fake_kill)

        await e._destroy_engine()

        touched = {pid for pid, _ in kills}
        assert touched == {11111}, f"must only touch own pid, touched {touched}"
        assert scans["n"] == 0, "destroy must not scan active_children()"
        assert e._child_pids == set(), "own pid set must be cleared after destroy"
    asyncio.run(run())


# --- 2) create records only the delta as own children ------------------------

def test_create_records_child_delta(monkeypatch):
    async def run():
        import multiprocessing
        calls = {"n": 0}
        def fake_children():
            calls["n"] += 1
            if calls["n"] == 1:           # before create: a sibling's core
                return [_FakeChild(1)]
            return [_FakeChild(1), _FakeChild(2)]  # after: sibling + own
        monkeypatch.setattr(multiprocessing, "active_children", fake_children)

        e = HardwareFake()
        e._model = "m"
        await e._create_engine()
        assert e._child_pids == {2}, f"delta attribution wrong: {e._child_pids}"
    asyncio.run(run())


# --- 3) EngineDeadError → DEAD + retryable error + recovery scheduled --------

def test_engine_dead_marks_dead_and_raises_retryable():
    async def run():
        e = HardwareFake()
        await e.start("m")
        assert e.is_available()

        recovered = asyncio.Event()
        async def fake_recover():
            recovered.set()
            e._recovering = False
        e._recover_dead_engine = fake_recover

        class DeadEngine:
            def generate(self, *_a, **_k):
                async def gen():
                    raise EngineDeadError("core crashed")
                    yield  # pragma: no cover
                return gen()
        e._engine = DeadEngine()

        with pytest.raises(RuntimeError, match="retry"):
            await e.generate("hi", max_tokens=4)

        assert e._state == EngineState.DEAD
        assert not e.is_available()
        await asyncio.wait_for(recovered.wait(), timeout=2)
        assert e._active_requests == 0, "active counter must unwind on failure"
    asyncio.run(run())


# --- 4) pool excludes dead engines; aggregate reports DEAD when all dead -----

def test_pool_excludes_dead_engine_and_reports_dead():
    async def run():
        e1, e2 = MockEngine(gpu_id=0), MockEngine(gpu_id=1)
        pool = EnginePool(mode="multi_instance", engines=[e1, e2])
        await pool.start("m")

        e1._state = EngineState.DEAD
        # Routing must always land on the surviving engine.
        for _ in range(4):
            assert pool._pick_engine() is e2
        # One alive → pool still reports RUNNING (degraded but serving).
        assert pool.status().state == EngineState.RUNNING

        e2._state = EngineState.DEAD
        # All dead → pool reports DEAD and refuses to route.
        assert pool.status().state == EngineState.DEAD
        with pytest.raises(RuntimeError):
            pool._pick_engine()
    asyncio.run(run())


# --- 5) switch_model drains in-flight work before pausing --------------------

def test_switch_model_drains_before_pause():
    async def run():
        e = HardwareFake()
        await e.start("old")
        e._active_requests = 1  # simulate one in-flight request

        order = []
        real_pause, real_resume = e.pause, e.resume
        async def rec_pause(*a, **k):
            order.append(("pause", e._active_requests))
            await real_pause(*a, **k)
        async def rec_resume(*a, **k):
            order.append(("resume", None))
            await real_resume(*a, **k)
        e.pause, e.resume = rec_pause, rec_resume

        task = asyncio.create_task(e.switch_model("new"))
        await asyncio.sleep(0.15)
        # Mid-drain: still RUNNING (in-flight unharmed) but refusing new work.
        assert e._state == EngineState.RUNNING
        assert not e.is_available()
        assert order == [], "must not pause while a request is in flight"

        e._active_requests = 0  # in-flight request completes
        await asyncio.wait_for(task, timeout=5)

        assert order and order[0] == ("pause", 0), \
            "pause must happen only after the drain emptied in-flight work"
        assert e.current_model == "new"
        assert e._state == EngineState.RUNNING
        assert e.is_available(), "_accepting must be restored after the switch"
    asyncio.run(run())


# --- 6) pool switch aborts when a mining yield arrives mid-switch ------------

def test_pool_switch_aborts_on_pause_request():
    async def run():
        e1, e2 = MockEngine(gpu_id=0), MockEngine(gpu_id=1)
        pool = EnginePool(mode="multi_instance", engines=[e1, e2])
        await pool.start("old")

        pool.request_pause()  # mining yield signal
        await pool.switch_model("new")
        assert e1.current_model == "old", "switch must abort before engine 1"
        assert e2.current_model == "old"
    asyncio.run(run())


# --- 7) soft yield (NORMAL): in-flight finishes, new requests rejected -------

def test_soft_pause_lets_inflight_finish_and_rejects_new():
    """The user-visible contract of a soft (WindowPoSt) yield: requests already
    running keep running to completion; new arrivals are refused. The old code
    flipped to UNLOADING before draining, which aborted everything in-flight."""
    async def run():
        from src.inference.engine import YieldUrgency, GenerateResult

        e = HardwareFake()
        await e.start("m")

        # A fake core whose generation takes several ticks to finish.
        class SlowEngine:
            def generate(self, *_a, **_k):
                async def gen():
                    class Out:
                        text = "done"
                        token_ids = [1, 2]
                        finish_reason = "stop"
                    class RO:
                        outputs = [Out()]
                        prompt_token_ids = [1]
                    for _ in range(5):
                        await asyncio.sleep(0.05)
                        yield RO()
                return gen()
            async def abort(self, *_a):
                pass
            def shutdown(self):
                pass
        e._engine = SlowEngine()

        inflight = asyncio.create_task(e.generate("hi", max_tokens=4))
        await asyncio.sleep(0.06)  # let it take its first tick

        pause_task = asyncio.create_task(e.pause(YieldUrgency.NORMAL))
        await asyncio.sleep(0.05)

        # Mid-drain: engine still RUNNING (in-flight unharmed) but not accepting.
        assert e._state == EngineState.RUNNING
        assert not e.is_available()
        with pytest.raises(RuntimeError):
            await e.generate("new request", max_tokens=4)

        # The in-flight request must COMPLETE, not abort.
        result = await asyncio.wait_for(inflight, timeout=5)
        assert isinstance(result, GenerateResult) and result.text == "done"

        await asyncio.wait_for(pause_task, timeout=5)
        assert e._state == EngineState.PAUSED
    asyncio.run(run())


# --- 8) immediate yield (WinningPoSt) still aborts right away ----------------

def test_immediate_pause_aborts_without_drain():
    async def run():
        from src.inference.engine import YieldUrgency

        e = HardwareFake()
        await e.start("m")
        e._engine = types.SimpleNamespace(shutdown=lambda: None)
        e._active_requests = 1  # simulated stuck in-flight request

        # Must NOT wait for the in-flight request — block production is urgent.
        await asyncio.wait_for(e.pause(YieldUrgency.IMMEDIATE), timeout=2)
        assert e._state == EngineState.PAUSED
    asyncio.run(run())


# --- 9) immediate pause arriving mid-soft-drain takes over -------------------

def test_immediate_pause_overrides_soft_drain():
    async def run():
        from src.inference.engine import YieldUrgency

        e = HardwareFake()
        await e.start("m")
        e._engine = types.SimpleNamespace(shutdown=lambda: None)
        e._active_requests = 1

        soft = asyncio.create_task(e.pause(YieldUrgency.NORMAL))
        await asyncio.sleep(0.05)
        assert e._state == EngineState.RUNNING  # draining, not yet unloading

        # WinningPoSt arrives: immediate pause must take over NOW.
        hard = asyncio.create_task(e.pause(YieldUrgency.IMMEDIATE))
        await asyncio.sleep(0.05)
        assert e._state in (EngineState.UNLOADING, EngineState.PAUSED)

        e._active_requests = 0  # the aborted in-flight unwinds
        await asyncio.wait_for(asyncio.gather(soft, hard), timeout=5)
        assert e._state == EngineState.PAUSED
        assert e._accepting, "accepting must be re-armed for the next resume"
    asyncio.run(run())


# --- 10) resume() on a DEAD engine must not silently no-op into recovery ----

def test_resume_on_dead_engine_does_not_clobber_recovery():
    """A yield→resume can dispatch resume() while a crashed engine is mid-recovery.
    resume() must leave the DEAD state for the recovery task to finish (and warn),
    NOT transition it or pretend success in a way that aborts recovery."""
    async def run():
        e = HardwareFake()
        await e.start("m")
        e._state = EngineState.DEAD  # core crashed, recovery task reloading

        await asyncio.wait_for(e.resume(), timeout=2)

        # resume() must not have flipped a DEAD engine to LOADING/RUNNING itself;
        # recovery owns that transition.
        assert e._state == EngineState.DEAD
    asyncio.run(run())


# --- 11) pool reports LOADING for the WHOLE multi-instance switch -------------

def test_pool_reports_loading_during_switch():
    """Multi-instance switch is sequential; not-yet-switched engines keep
    reporting RUNNING and loaded_model flips after engine 0. The pool must report
    LOADING for the whole switch so the gateway excludes the worker until every
    engine is done (else requests routed mid-switch hang on a reloading engine)."""
    async def run():
        e1, e2 = MockEngine(gpu_id=0), MockEngine(gpu_id=1)
        pool = EnginePool(mode="multi_instance", engines=[e1, e2])
        await pool.start("old")
        assert pool.status().state == EngineState.RUNNING

        # Both engines still report RUNNING, but a switch is in progress → LOADING.
        pool._switching = True
        assert pool.status().state == EngineState.LOADING
        assert not pool.is_available() or pool.status().state == EngineState.LOADING

        pool._switching = False
        assert pool.status().state == EngineState.RUNNING
    asyncio.run(run())


def test_switch_model_holds_loading_until_all_engines_done():
    async def run():
        e1, e2 = MockEngine(gpu_id=0), MockEngine(gpu_id=1)
        pool = EnginePool(mode="multi_instance", engines=[e1, e2])
        await pool.start("old")

        release = asyncio.Event()
        orig = e2.switch_model
        async def slow(m):
            await release.wait()
            await orig(m)
        e2.switch_model = slow

        task = asyncio.create_task(pool.switch_model("new"))
        await asyncio.sleep(0.05)
        # Engine 1 already switched (RUNNING on "new"); engine 2 still pending.
        # Pool must still report LOADING — not RUNNING.
        assert pool._switching is True
        assert pool.status().state == EngineState.LOADING

        release.set()
        await asyncio.wait_for(task, timeout=5)
        assert pool._switching is False
        assert pool.status().state == EngineState.RUNNING
        assert pool.current_model == "new"
    asyncio.run(run())


def test_switch_model_clears_switching_flag_on_abort():
    """A mining yield mid-switch aborts the loop; the finally must still clear the
    flag so the worker isn't stuck reporting LOADING forever."""
    async def run():
        e1, e2 = MockEngine(gpu_id=0), MockEngine(gpu_id=1)
        pool = EnginePool(mode="multi_instance", engines=[e1, e2])
        await pool.start("old")
        pool.request_pause()  # _pause_requested = True → switch aborts before engine 1
        await pool.switch_model("new")
        assert pool._switching is False
    asyncio.run(run())


# --- 9) a wedged/dead core that stops producing output fails fast (no hang) --

def test_stalled_engine_fails_fast_instead_of_hanging(monkeypatch):
    """Soak finding (#46): when the EngineCore is SIGKILL'd mid-generation the vLLM
    result stream stops yielding without promptly raising, so the request used to hang
    until the client's own timeout (~120s), giving the gateway no chance to retry.
    _iter_outputs bounds each token await: a stall raises a retryable RuntimeError (→503)
    and aborts the request, so the worker fails fast instead of holding the connection."""
    import src.inference.engine as eng
    monkeypatch.setattr(eng, "_GEN_STALL_TIMEOUT", 0.05)  # make the stall guard fire fast

    async def run():
        e = HardwareFake()
        await e.start("m")

        aborted = {"called": False}

        class HangEngine:
            def generate(self, *_a, **_k):
                async def gen():
                    await asyncio.sleep(3600)  # dead core: never produces a token
                    yield None
                return gen()

            async def abort(self, *_a):
                aborted["called"] = True

            def shutdown(self):
                pass

        e._engine = HangEngine()

        # Must raise (fast) rather than hang; the outer wait_for guards the test itself.
        with pytest.raises(RuntimeError, match="stalled"):
            await asyncio.wait_for(e.generate("hi", max_tokens=4), timeout=5)
        assert aborted["called"], "a stalled request must be aborted so the core can reclaim it"

    asyncio.run(run())
