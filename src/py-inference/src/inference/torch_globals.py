"""Keep in-process model loads apart from in-process model calls.

diffusers and transformers change torch's PROCESS-WIDE default dtype while they load
a model (`torch.set_default_dtype(float16)`, restored when the load returns). A model
call on another thread in that window builds its tensors in the wrong dtype: on a test worker,
Kokoro synthesis on card 6 failed with "expected scalar type Half but found Float"
for every request that overlapped an SDXL <-> SSD-1B switch on card 4 (17 of 87, and
19 of 60 against a thread flipping the default in a standalone container).

Loads used to happen only while the whole worker resumed, with no requests running,
so the window never met a call; a switch inside one pool while the others serve opens
it. vLLM pools load in their own engine-core process and are not affected.

A readers-writer lock: model calls are readers and run concurrently with each other
(each engine still serialises its own calls); a load is a writer and runs alone,
waiting for calls in flight and holding new ones back until it returns. Writers are
preferred, so a steady stream of calls cannot starve a switch. A load is seconds, so
the cost is a short pause of the other in-process pools during a switch.
"""
from __future__ import annotations

import contextlib
import threading

_cond = threading.Condition()
_readers = 0
_writer = False
_writers_waiting = 0


@contextlib.contextmanager
def model_call():
    """Held around a model call (render, transcription, synthesis) in a worker thread."""
    global _readers
    with _cond:
        while _writer or _writers_waiting:
            _cond.wait()
        _readers += 1
    try:
        yield
    finally:
        with _cond:
            _readers -= 1
            if _readers == 0:
                _cond.notify_all()


@contextlib.contextmanager
def model_load():
    """Held around an in-process model load (from_pretrained, KModel, WhisperModel)."""
    global _writer, _writers_waiting
    with _cond:
        _writers_waiting += 1
        while _writer or _readers:
            _cond.wait()
        _writers_waiting -= 1
        _writer = True
    try:
        yield
    finally:
        with _cond:
            _writer = False
            _cond.notify_all()


def guarded_call(fn, *args, **kwargs):
    with model_call():
        return fn(*args, **kwargs)


def guarded_load(fn, *args, **kwargs):
    with model_load():
        return fn(*args, **kwargs)
