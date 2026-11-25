"""A model switch whose new model fails to load must leave the pool serving its old one.

The vLLM engine rolls back; the image and speech engines unloaded the old model, failed
on the new one and stayed down (pointing at the model that cannot load) until a mining
yield and resume happened to reload it.
"""
import asyncio

import pytest

from src.inference.diffusion_engine import DiffusionEngine
from src.inference.engine import EngineState
from src.inference.speech_engine import AsrEngine

OLD, BAD = "/models/old-model", "/models/broken-model"


def test_failed_image_switch_rolls_back_to_the_old_model():
    async def run():
        e = DiffusionEngine(device_ids=[0], max_images_per_request=1)
        loads = []

        async def load():
            loads.append(e._model)
            if e._model == BAD:
                raise OSError("unet weights missing")
            e._pipe = object()

        e._load_pipeline = load
        await e.start(OLD)
        with pytest.raises(OSError):
            await e.switch_model(BAD)
        assert e.current_model == OLD
        assert e.status().state is EngineState.RUNNING and e.is_available()
        assert loads == [OLD, BAD, OLD]
    asyncio.run(run())


def test_failed_speech_switch_rolls_back_to_the_old_model():
    async def run():
        e = AsrEngine(device_ids=[5])

        def load(path):
            if path == BAD:
                raise RuntimeError("model.bin is not a CTranslate2 model")
            return object()

        e._load = load
        await e.start(OLD)
        with pytest.raises(RuntimeError):
            await e.switch_model(BAD)
        assert e.current_model == OLD
        assert e.status().state is EngineState.RUNNING and e.is_available()
    asyncio.run(run())


def test_rollback_that_also_fails_leaves_the_engine_retryable():
    async def run():
        e = AsrEngine(device_ids=[5])
        ok = {"old": True}

        def load(path):
            if path == BAD or not ok["old"]:
                raise RuntimeError("load failed")
            return object()

        e._load = load
        await e.start(OLD)
        ok["old"] = False
        with pytest.raises(RuntimeError):
            await e.switch_model(BAD)
        # Down, but PAUSED on the old model: the next resume retries it.
        assert e.status().state is EngineState.PAUSED
        assert e.current_model == OLD
        ok["old"] = True
        await e.resume()
        assert e.status().state is EngineState.RUNNING
    asyncio.run(run())
