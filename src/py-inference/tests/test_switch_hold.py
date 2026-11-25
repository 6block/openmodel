"""A model switch must not pull a model out from under requests that already got it.

Found on a test worker (32B and VL-8B sharing one pool, four interleaved requests): request A
switched to VL; request B, queued behind that switch, immediately switched back to the
32B before A had generated anything, so A met an unavailable engine and answered 503.
Each request must keep the model it obtained until its response — including a whole
SSE stream — has been sent; a switch waits for that.
"""
import asyncio
import threading

from fastapi.testclient import TestClient

from src.inference.api_server import create_app
from src.inference.engine import MockEngine


class _Manager:
    async def ensure_model(self, model_id):
        return model_id


class _SlowSwitchEngine(MockEngine):
    """Like the real pool: while switching the engine is unavailable, and a request
    whose model was swapped out during its generation fails."""

    async def switch_model(self, new_model_path):
        from src.inference.engine import EngineState
        self._state = EngineState.LOADING
        await asyncio.sleep(0.3)          # unload + load
        self._model = new_model_path
        self._state = EngineState.RUNNING

    async def generate(self, prompt, **kw):
        model = self._model
        r = await super().generate(prompt, **kw)
        if self._model != model:
            raise RuntimeError("Engine not available — model switched during generation")
        return r

    async def generate_stream(self, prompt, **kw):
        model = self._model
        async for ch in super().generate_stream(prompt, **kw):
            if self._model != model:
                raise RuntimeError("Engine not available — model switched during generation")
            yield ch


def _client():
    # One event loop for every request, as under uvicorn: TestClient used outside a
    # `with` block runs each request on a fresh loop, which a shared asyncio lock
    # cannot span.
    e = _SlowSwitchEngine(latency_sec=0.4)
    asyncio.run(e.start("model-a"))
    return TestClient(create_app(e, model_manager=_Manager())), e


def test_interleaved_requests_for_two_models_all_succeed():
    c0, e = _client()
    c = c0.__enter__()
    out = {}

    def call(i, model, stream=False):
        body = {"model": model, "messages": [{"role": "user", "content": "x"}], "max_tokens": 4, "stream": stream}
        if stream:
            with c.stream("POST", "/v1/chat/completions", json=body) as r:
                lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
                out[i] = (r.status_code, lines[-1] if lines else "", any('"error"' in ln for ln in lines))
        else:
            r = c.post("/v1/chat/completions", json=body)
            out[i] = (r.status_code, "", False)

    th = [threading.Thread(target=call, args=(i, m, i % 3 == 2))
          for i, m in enumerate(["model-b", "model-a", "model-b", "model-a", "model-b", "model-a"])]
    for t in th:
        t.start()
    for t in th:
        t.join()
    c0.__exit__(None, None, None)
    bad = {i: v for i, v in out.items() if v[0] != 200 or v[2]}
    assert not bad, f"requests failed while the pool switched under them: {bad}"


def test_hold_is_released_after_each_response():
    c0, e = _client()
    c = c0.__enter__()
    for m in ["model-b", "model-a", "model-b"]:
        r = c.post("/v1/chat/completions", json={"model": m, "messages": [{"role": "user", "content": "x"}], "max_tokens": 4})
        assert r.status_code == 200
    assert e.current_model == "model-b"
    c0.__exit__(None, None, None)


def test_switch_of_one_pool_does_not_block_another():
    """A multi-pool worker: switching the text pool must not delay a request for a
    resident model in another pool."""
    from src.inference.multipool import MultiPoolEngine, Pool
    from src.inference.engine_pool import EnginePool
    import time

    text = _SlowSwitchEngine(latency_sec=0.05)
    text.switch_delay = 1.5
    other = MockEngine(latency_sec=0.05)

    async def boot():
        await text.start("model-a")
        await other.start("model-x")
    asyncio.run(boot())

    async def slow_switch(new_model_path, _t=text):
        from src.inference.engine import EngineState
        _t._state = EngineState.LOADING
        await asyncio.sleep(1.5)
        _t._model = new_model_path
        _t._state = EngineState.RUNNING
    text.switch_model = slow_switch
    m = MultiPoolEngine([
        Pool(name="text", engine=EnginePool("tensor_parallel", [text]), devices=[0], supported_models=["model-a", "model-b"]),
        Pool(name="other", engine=EnginePool("tensor_parallel", [other]), devices=[1], supported_models=["model-x"]),
    ])
    c0 = TestClient(create_app(m, model_manager=_Manager()))
    c = c0.__enter__()
    out = {}

    def switcher():
        out["switch"] = c.post("/v1/chat/completions", json={"model": "model-b", "messages": [{"role": "user", "content": "x"}], "max_tokens": 4}).status_code

    th = threading.Thread(target=switcher)
    th.start()
    time.sleep(0.3)
    t0 = time.time()
    r = c.post("/v1/chat/completions", json={"model": "model-x", "messages": [{"role": "user", "content": "x"}], "max_tokens": 4})
    dt = time.time() - t0
    th.join()
    c0.__exit__(None, None, None)
    assert r.status_code == 200 and out["switch"] == 200
    assert dt < 0.8, f"request for another pool waited {dt:.2f}s behind the text pool's switch"
