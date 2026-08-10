"""SP self-registration token flow on the inference side: the listener learns the
per-worker gateway token from the scheduler (gRPC), the token store keeps the
last two through a rotation, and the API server accepts static AND dynamic
tokens — enforcing auth as soon as either exists."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import create_app
from src.inference.engine import MockEngine
from src.scheduler_client.token_store import GatewayTokenStore


@pytest.fixture
def engine():
    e = MockEngine(latency_sec=0.0)
    asyncio.run(e.start("test-model"))
    return e


def _chat(client, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}]},
                       headers=headers)


class TestGatewayTokenStore:
    def test_empty_updates_ignored(self):
        s = GatewayTokenStore()
        s.update("")
        assert s.accepted() == []

    def test_keeps_last_two_across_rotation(self):
        s = GatewayTokenStore()
        s.update("wk-a")
        assert s.accepted() == ["wk-a"]
        s.update("wk-b")  # rotation: both valid during handover
        assert s.accepted() == ["wk-a", "wk-b"]
        s.update("wk-c")  # oldest evicted
        assert s.accepted() == ["wk-b", "wk-c"]

    def test_repeated_token_is_stable(self):
        s = GatewayTokenStore()
        s.update("wk-a")
        s.update("wk-b")
        for _ in range(5):
            s.update("wk-b")  # every 3s poll re-advertises the same token
        assert s.accepted() == ["wk-a", "wk-b"]


class TestDynamicAuth:
    def test_dynamic_token_accepted(self, engine):
        store = GatewayTokenStore()
        store.update("wk-dynamic")
        app = create_app(engine, api_token="", token_store=store)
        client = TestClient(app)
        assert _chat(client, "wk-dynamic").status_code == 200
        assert _chat(client, "wrong").status_code == 401
        assert _chat(client).status_code == 401  # no header

    def test_dynamic_alone_enables_auth(self, engine):
        """With no static token, auth turns ON the moment a dynamic token exists."""
        store = GatewayTokenStore()
        app = create_app(engine, api_token="", token_store=store)
        client = TestClient(app)
        # Store empty → trusted-LAN mode, open (backward compatible).
        assert _chat(client).status_code == 200
        store.update("wk-x")
        assert _chat(client).status_code == 401
        assert _chat(client, "wk-x").status_code == 200

    def test_static_and_dynamic_both_accepted(self, engine):
        store = GatewayTokenStore()
        store.update("wk-old")
        store.update("wk-new")
        app = create_app(engine, api_token="static-secret", token_store=store)
        client = TestClient(app)
        for tok in ("static-secret", "wk-old", "wk-new"):
            assert _chat(client, tok).status_code == 200, tok
        assert _chat(client, "wk-evicted").status_code == 401

    def test_rotation_handover(self, engine):
        """Both the pre- and post-rotation tokens work during the handover window."""
        store = GatewayTokenStore()
        store.update("wk-v1")
        app = create_app(engine, api_token="", token_store=store)
        client = TestClient(app)
        assert _chat(client, "wk-v1").status_code == 200
        store.update("wk-v2")  # scheduler re-registered → rotated
        assert _chat(client, "wk-v1").status_code == 200  # in-flight window
        assert _chat(client, "wk-v2").status_code == 200


class TestListenerFeedsStore:
    def test_poll_updates_store(self):
        """The listener pushes each poll's advertised token into the store, and
        rotations propagate on the next poll."""
        import threading

        from conftest import FakeSchedulerClient
        from src.scheduler_client.listener import SchedulerListener
        from src.inference.engine import EngineState, EngineStatus, YieldUrgency

        class StatusEngine:
            async def pause(self, urgency=YieldUrgency.NORMAL):
                pass

            async def resume(self):
                pass

            def request_pause(self):
                pass

            def status(self):
                return EngineStatus(state=EngineState.RUNNING, active_requests=0,
                                    loaded_model="m", gpu_id=0)

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            client = FakeSchedulerClient()
            client.gateway_auth_token = "wk-from-scheduler"
            store = GatewayTokenStore()
            listener = SchedulerListener(client, StatusEngine(), main_loop=loop,
                                         token_store=store)
            listener._poll()
            assert store.accepted() == ["wk-from-scheduler"]

            client.gateway_auth_token = "wk-rotated"
            listener._poll()
            assert store.accepted() == ["wk-from-scheduler", "wk-rotated"]
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2)
            loop.close()
