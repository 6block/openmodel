"""Shared test fixtures and fakes for py-inference unit tests.

These tests run WITHOUT a GPU or vLLM. The vLLM-touching methods of VLLMEngine
(_create_engine / _destroy_engine / _wait_for_vram_release) are overridden by
FakeVLLMEngine so the real pause/resume/start state machine can be exercised.
"""
import os
import sys
import types

# Make `src` importable as a top-level package path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The generated sidecar_pb2 is built for the server's protobuf runtime and may
# not parse under the local protobuf version. The listener only needs the
# SchedulerClient *type* (for an annotation), so stub the grpc_client module to
# keep these unit tests free of the protobuf/gRPC runtime.
_grpc_stub = types.ModuleType("src.scheduler_client.grpc_client")


class SchedulerClient:  # minimal stand-in
    pass


_grpc_stub.SchedulerClient = SchedulerClient
sys.modules.setdefault("src.scheduler_client.grpc_client", _grpc_stub)


class FakeSchedule:
    """Stand-in for the gRPC ScheduleResponse used by the listener."""
    def __init__(self, state: int, message: str = ""):
        self.state = state
        self.message = message


class FakeSchedulerClient:
    """Controllable scheduler client for listener tests."""
    def __init__(self):
        self._state = 1  # AVAILABLE
        self.raise_on_get = False
        self.reported = []  # status reports captured (P5)

    def set_state(self, state: int):
        self._state = state

    def get_gpu_schedule(self):
        if self.raise_on_get:
            raise ConnectionError("scheduler unreachable")
        return FakeSchedule(self._state)

    def report_status(self, **kwargs):
        self.reported.append(kwargs)
