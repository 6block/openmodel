from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GpuState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GPU_STATE_UNKNOWN: _ClassVar[GpuState]
    GPU_STATE_AVAILABLE: _ClassVar[GpuState]
    GPU_STATE_YIELDING: _ClassVar[GpuState]
    GPU_STATE_WINDOW_POST: _ClassVar[GpuState]
    GPU_STATE_WINNING_POST: _ClassVar[GpuState]

class YieldUrgency(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    YIELD_URGENCY_NORMAL: _ClassVar[YieldUrgency]
    YIELD_URGENCY_HIGH: _ClassVar[YieldUrgency]
    YIELD_URGENCY_IMMEDIATE: _ClassVar[YieldUrgency]

class YieldReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    YIELD_REASON_UNKNOWN: _ClassVar[YieldReason]
    YIELD_REASON_WINDOW_POST_APPROACHING: _ClassVar[YieldReason]
    YIELD_REASON_WINDOW_POST_ACTIVE: _ClassVar[YieldReason]
    YIELD_REASON_WINNING_POST: _ClassVar[YieldReason]
    YIELD_REASON_LOTUS_DISCONNECTED: _ClassVar[YieldReason]
    YIELD_REASON_MANUAL: _ClassVar[YieldReason]
    YIELD_REASON_RESUME: _ClassVar[YieldReason]
GPU_STATE_UNKNOWN: GpuState
GPU_STATE_AVAILABLE: GpuState
GPU_STATE_YIELDING: GpuState
GPU_STATE_WINDOW_POST: GpuState
GPU_STATE_WINNING_POST: GpuState
YIELD_URGENCY_NORMAL: YieldUrgency
YIELD_URGENCY_HIGH: YieldUrgency
YIELD_URGENCY_IMMEDIATE: YieldUrgency
YIELD_REASON_UNKNOWN: YieldReason
YIELD_REASON_WINDOW_POST_APPROACHING: YieldReason
YIELD_REASON_WINDOW_POST_ACTIVE: YieldReason
YIELD_REASON_WINNING_POST: YieldReason
YIELD_REASON_LOTUS_DISCONNECTED: YieldReason
YIELD_REASON_MANUAL: YieldReason
YIELD_REASON_RESUME: YieldReason

class ScheduleEvent(_message.Message):
    __slots__ = ("state", "urgency", "seconds_until_deadline", "current_epoch", "deadline_open_epoch", "reason", "message")
    STATE_FIELD_NUMBER: _ClassVar[int]
    URGENCY_FIELD_NUMBER: _ClassVar[int]
    SECONDS_UNTIL_DEADLINE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_OPEN_EPOCH_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    state: GpuState
    urgency: YieldUrgency
    seconds_until_deadline: int
    current_epoch: int
    deadline_open_epoch: int
    reason: YieldReason
    message: str
    def __init__(self, state: _Optional[_Union[GpuState, str]] = ..., urgency: _Optional[_Union[YieldUrgency, str]] = ..., seconds_until_deadline: _Optional[int] = ..., current_epoch: _Optional[int] = ..., deadline_open_epoch: _Optional[int] = ..., reason: _Optional[_Union[YieldReason, str]] = ..., message: _Optional[str] = ...) -> None: ...

class InferenceStatusReport(_message.Message):
    __slots__ = ("is_running", "active_requests", "gpu_utilization_pct", "loaded_model", "timestamp_unix")
    IS_RUNNING_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_PCT_FIELD_NUMBER: _ClassVar[int]
    LOADED_MODEL_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_UNIX_FIELD_NUMBER: _ClassVar[int]
    is_running: bool
    active_requests: int
    gpu_utilization_pct: float
    loaded_model: str
    timestamp_unix: int
    def __init__(self, is_running: bool = ..., active_requests: _Optional[int] = ..., gpu_utilization_pct: _Optional[float] = ..., loaded_model: _Optional[str] = ..., timestamp_unix: _Optional[int] = ...) -> None: ...

class ScheduleRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ScheduleResponse(_message.Message):
    __slots__ = ("current_state", "seconds_until_next_change", "message")
    CURRENT_STATE_FIELD_NUMBER: _ClassVar[int]
    SECONDS_UNTIL_NEXT_CHANGE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    current_state: GpuState
    seconds_until_next_change: int
    message: str
    def __init__(self, current_state: _Optional[_Union[GpuState, str]] = ..., seconds_until_next_change: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
