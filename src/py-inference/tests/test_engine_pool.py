"""Unit tests for EnginePool aggregation / load-balancing / construction (no GPU)."""
import pytest

from src.inference.engine import EngineState, EngineStatus
from src.inference.engine_pool import EnginePool, create_engine_pool


class FakeEngine:
    def __init__(self, state=EngineState.RUNNING, active=0, model="m", gpu_id=0):
        self._s = state
        self._active = active
        self._model = model
        self._gpu_id = gpu_id

    def status(self):
        return EngineStatus(state=self._s, active_requests=self._active,
                            loaded_model=self._model, gpu_id=self._gpu_id)

    def is_available(self):
        return self._s == EngineState.RUNNING

    @property
    def current_model(self):
        return self._model


def pool(*engines, mode="multi_instance", lb="round_robin"):
    return EnginePool(mode=mode, engines=list(engines), lb_strategy=lb)


def test_status_all_running_sums_active():
    s = pool(FakeEngine(active=1), FakeEngine(active=2)).status()
    assert s.state == EngineState.RUNNING
    assert s.active_requests == 3


def test_status_all_paused():
    assert pool(FakeEngine(state=EngineState.PAUSED), FakeEngine(state=EngineState.PAUSED)).status().state == EngineState.PAUSED


def test_status_all_stopped():
    assert pool(FakeEngine(state=EngineState.STOPPED), FakeEngine(state=EngineState.STOPPED)).status().state == EngineState.STOPPED


def test_status_mixed_reports_running():
    # one running + one paused → still serveable → RUNNING
    assert pool(FakeEngine(state=EngineState.RUNNING), FakeEngine(state=EngineState.PAUSED)).status().state == EngineState.RUNNING


def test_status_tensor_parallel_delegates():
    s = pool(FakeEngine(state=EngineState.PAUSED, active=5), mode="tensor_parallel").status()
    assert s.state == EngineState.PAUSED and s.active_requests == 5


def test_pick_engine_round_robin():
    p = pool(FakeEngine(gpu_id=0), FakeEngine(gpu_id=1), FakeEngine(gpu_id=2), lb="round_robin")
    picks = [p._pick_engine()._gpu_id for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2]


def test_pick_engine_least_busy():
    p = pool(FakeEngine(gpu_id=0, active=5), FakeEngine(gpu_id=1, active=1), FakeEngine(gpu_id=2, active=3), lb="least_busy")
    assert p._pick_engine()._gpu_id == 1


def test_pick_engine_skips_unavailable():
    p = pool(FakeEngine(gpu_id=0, state=EngineState.PAUSED), FakeEngine(gpu_id=1, state=EngineState.RUNNING))
    assert p._pick_engine()._gpu_id == 1


def test_pick_engine_none_available_raises():
    with pytest.raises(RuntimeError):
        pool(FakeEngine(state=EngineState.PAUSED), FakeEngine(state=EngineState.PAUSED))._pick_engine()


def test_is_available_any():
    assert pool(FakeEngine(state=EngineState.PAUSED), FakeEngine(state=EngineState.RUNNING)).is_available()
    assert not pool(FakeEngine(state=EngineState.PAUSED)).is_available()


def test_current_model():
    assert pool(FakeEngine(model="foo")).current_model == "foo"
    assert EnginePool(mode="multi_instance", engines=[]).current_model == ""


def test_pick_engine_affinity_sticky():
    # With a session_key, repeated picks return the SAME engine (prefix-cache reuse),
    # instead of round-robining across the pool.
    p = pool(FakeEngine(gpu_id=0), FakeEngine(gpu_id=1), FakeEngine(gpu_id=2), lb="round_robin")
    first = p._pick_engine(session_key="s1")._gpu_id
    for _ in range(10):
        assert p._pick_engine(session_key="s1")._gpu_id == first
    # A different session is free to land on a different engine.
    second = p._pick_engine(session_key="s2")._gpu_id
    for _ in range(10):
        assert p._pick_engine(session_key="s2")._gpu_id == second


def test_pick_engine_affinity_busy_fallback():
    # If the pinned engine becomes much busier than the least-loaded one
    # (beyond the slack), routing falls back to balancing instead of sticking.
    e0, e1 = FakeEngine(gpu_id=0, active=0), FakeEngine(gpu_id=1, active=0)
    p = pool(e0, e1, lb="least_busy")
    pinned = p._pick_engine(session_key="s")._gpu_id
    assert pinned == 0  # tie → first
    e0._active = 10     # pinned engine now swamped
    assert p._pick_engine(session_key="s")._gpu_id == 1  # falls back to idle engine


def test_pick_engine_affinity_unavailable_fallback():
    # If the pinned engine goes away (paused), the session is re-pinned elsewhere.
    e0, e1 = FakeEngine(gpu_id=0), FakeEngine(gpu_id=1)
    p = pool(e0, e1, lb="least_busy")
    assert p._pick_engine(session_key="s")._gpu_id == 0
    e0._s = EngineState.PAUSED
    assert p._pick_engine(session_key="s")._gpu_id == 1


def test_pick_engine_no_session_key_is_round_robin():
    # No session key → unchanged balancing behaviour (backward compatible).
    p = pool(FakeEngine(gpu_id=0), FakeEngine(gpu_id=1), lb="round_robin")
    assert [p._pick_engine()._gpu_id for _ in range(4)] == [0, 1, 0, 1]


def _cfg(mode, count, devices, lb="round_robin"):
    from src.config import AppConfig, InferenceConfig, MultiGpuConfig, MultiInstanceConfig
    return AppConfig(mode="dev", inference=InferenceConfig(
        engine="mock",
        multi_gpu=MultiGpuConfig(mode=mode, gpu_count=count, device_ids=devices),
        multi_instance=MultiInstanceConfig(load_balancer=lb),
    ))


def test_create_engine_pool_tensor_parallel():
    p = create_engine_pool(_cfg("tensor_parallel", 4, [0, 1, 2, 3]))
    assert p.mode == "tensor_parallel"
    assert p.engine_count == 1  # one engine spans all GPUs


def test_create_engine_pool_multi_instance():
    p = create_engine_pool(_cfg("multi_instance", 3, [0, 1, 2], lb="least_busy"))
    assert p.mode == "multi_instance"
    assert p.engine_count == 3  # one engine per device


def test_create_engine_pool_unknown_mode_raises():
    with pytest.raises(ValueError):
        create_engine_pool(_cfg("bogus", 1, [0]))
