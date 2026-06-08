"""gRPC client for communicating with the Go scheduler."""

import logging
import time
from dataclasses import dataclass

import grpc

from . import sidecar_pb2
from . import sidecar_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass
class GpuSchedule:
    state: int  # GpuState enum value
    seconds_until_next_change: int
    message: str

    @property
    def is_available(self) -> bool:
        return self.state == sidecar_pb2.GPU_STATE_AVAILABLE

    @property
    def state_name(self) -> str:
        return sidecar_pb2.GpuState.Name(self.state)


class SchedulerClient:
    """gRPC client to communicate with the Go scheduler."""

    def __init__(self, addr: str):
        self._addr = addr
        self._channel = None
        self._stub = None

    def connect(self):
        """Establish gRPC connection."""
        self._channel = grpc.insecure_channel(self._addr)
        self._stub = sidecar_pb2_grpc.SchedulerServiceStub(self._channel)
        logger.info("connected to scheduler", extra={"addr": self._addr})

    def close(self):
        """Close the gRPC connection."""
        if self._channel:
            self._channel.close()

    def get_gpu_schedule(self) -> GpuSchedule:
        """Poll the current GPU availability."""
        resp = self._stub.GetGpuSchedule(sidecar_pb2.ScheduleRequest())
        return GpuSchedule(
            state=resp.current_state,
            seconds_until_next_change=resp.seconds_until_next_change,
            message=resp.message,
        )

    def subscribe_events(self):
        """Subscribe to GPU schedule events (server-streaming)."""
        return self._stub.SubscribeScheduleEvents(sidecar_pb2.ScheduleRequest())

    def report_status(self, is_running: bool, active_requests: int,
                      gpu_utilization: float, model: str) -> GpuSchedule:
        """Report inference status to the scheduler."""
        report = sidecar_pb2.InferenceStatusReport(
            is_running=is_running,
            active_requests=active_requests,
            gpu_utilization_pct=gpu_utilization,
            loaded_model=model,
            timestamp_unix=int(time.time()),
        )
        resp = self._stub.ReportInferenceStatus(report)
        return GpuSchedule(
            state=resp.current_state,
            seconds_until_next_change=resp.seconds_until_next_change,
            message=resp.message,
        )
