"""Model loads in one pool must not overlap model calls in another (torch_globals).

diffusers/transformers flip torch's process-wide default dtype during a load; a Kokoro
call overlapping an SDXL <-> SSD-1B switch on a test worker failed with "expected scalar type Half
but found Float". The stand-in below flips a process-wide value the same way and the
call reads it, so an overlap is observable without torch.
"""
import asyncio
import threading
import time

from src.inference import torch_globals
from src.inference.diffusion_engine import DiffusionEngine
from src.inference.speech_engine import AsrEngine

GLOBAL = {"dtype": "float32"}


def flipping_load(path):
    GLOBAL["dtype"] = "float16"
    time.sleep(0.2)
    GLOBAL["dtype"] = "float32"
    return object()


def test_a_call_never_sees_the_dtype_a_load_sets_in_another_pool():
    async def run():
        asr = AsrEngine(device_ids=[5])
        seen = []

        class Model:
            def transcribe(self, *a, **k):
                seen.append(GLOBAL["dtype"])
                class Info:
                    duration, language = 1.0, "en"
                return iter(()), Info()

        asr._load = lambda path: Model()
        await asr.start("/models/whisper")

        img = DiffusionEngine(device_ids=[4], max_images_per_request=1)

        async def load():
            img._pipe = await asyncio.to_thread(torch_globals.guarded_load, flipping_load, img._model)

        img._load_pipeline = load
        await img.start("/models/sdxl")

        async def calls():
            for _ in range(40):
                await asr._run(asr._impl.transcribe, b"")
                await asyncio.sleep(0.005)

        await asyncio.gather(calls(), img.switch_model("/models/ssd-1b"), img.switch_model("/models/sdxl"))
        assert seen and set(seen) == {"float32"}, seen
    asyncio.run(run())


def test_calls_run_concurrently_and_a_load_waits_for_them():
    order = []
    started = threading.Barrier(2)

    def call(i):
        with torch_globals.model_call():
            started.wait(timeout=2)       # both calls inside at once
            time.sleep(0.1)
            order.append(f"call{i}")

    def load():
        time.sleep(0.03)                  # arrives while the calls run
        with torch_globals.model_load():
            order.append("load")

    ts = [threading.Thread(target=call, args=(i,)) for i in (1, 2)] + [threading.Thread(target=load)]
    [t.start() for t in ts]
    [t.join(timeout=3) for t in ts]
    assert order[-1] == "load" and sorted(order[:2]) == ["call1", "call2"], order


def test_a_waiting_load_holds_back_new_calls():
    order = []
    first_in = threading.Event()

    def first_call():
        with torch_globals.model_call():
            first_in.set()
            time.sleep(0.15)
            order.append("call-before")

    def load():
        first_in.wait()
        with torch_globals.model_load():
            order.append("load")

    def late_call():
        first_in.wait()
        time.sleep(0.05)                  # the load is already waiting
        with torch_globals.model_call():
            order.append("call-after")

    ts = [threading.Thread(target=f) for f in (first_call, load, late_call)]
    [t.start() for t in ts]
    [t.join(timeout=3) for t in ts]
    assert order == ["call-before", "load", "call-after"], order
