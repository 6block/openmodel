"""Entry point for the OpenModel inference service."""

import argparse
import asyncio
import logging
import os
import sys

import uvicorn

from .config import load_config
from .inference.api_server import create_app
from .inference.engine_pool import create_engine_pool
from .model.foc_mock import create_foc_client
from .model.manager import ModelManager
from .scheduler_client.grpc_client import SchedulerClient
from .scheduler_client.listener import SchedulerListener


def setup_logging(level: str, fmt: str):
    log_level = getattr(logging, level.upper(), logging.INFO)
    if fmt == "json":
        logging.basicConfig(
            level=log_level,
            format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


async def main():
    parser = argparse.ArgumentParser(description="OpenModel Inference Service")
    parser.add_argument(
        "--config", default="/etc/sidecar/sidecar-prod-test.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.format)

    logger = logging.getLogger("main")
    logger.info("starting inference sidecar", extra={"mode": cfg.mode})

    # Create inference engine pool (supports single or multi-GPU)
    engine = create_engine_pool(cfg)
    logger.info("engine pool created",
                extra={"mode": cfg.inference.multi_gpu.mode,
                       "gpu_count": cfg.inference.multi_gpu.gpu_count,
                       "device_ids": str(cfg.inference.multi_gpu.device_ids)})

    # Create FOC client and model manager
    foc_client = create_foc_client({
        "backend": cfg.foc.backend,
        "mock": {
            "model_registry": cfg.foc.mock_model_registry,
            "simulate_delay": cfg.foc.simulate_delay,
        },
        "real": {
            "bridge_url": cfg.foc.real_bridge_url,
            "piece_cid_registry": cfg.foc.real_piece_cid_registry,
        },
    })
    # Cache dir: use MODEL_CACHE_DIR env var (set to /models in Docker, where the
    # shared volume is mounted in both inference and foc-bridge containers).
    # Fall back to ~/openmodel/models for bare-metal local development.
    model_cache_dir = os.environ.get("MODEL_CACHE_DIR",
                                     os.path.expanduser("~/openmodel/models"))
    model_manager = ModelManager(foc_client, cache_dir=model_cache_dir)
    logger.info("model cache dir: %s", model_cache_dir)

    # Connect to scheduler BEFORE loading model — check if GPU is available
    scheduler_client = SchedulerClient(cfg.grpc.scheduler_addr)
    listener = None
    gpu_available = False
    try:
        scheduler_client.connect()
        schedule = scheduler_client.get_gpu_schedule()
        gpu_available = schedule.is_available
        if not gpu_available:
            logger.info("GPU not available at startup (%s), deferring model load",
                        schedule.state_name)
    except Exception as e:
        logger.warning("could not connect to scheduler, deferring model load for safety",
                       extra={"error": str(e)})

    # Load model only if GPU is available; otherwise start paused
    model_path = await model_manager.ensure_model(cfg.inference.model)
    if gpu_available:
        await engine.start(model_path)
        logger.info("engine started", extra={"model": model_path})
    else:
        await engine.start_paused(model_path)
        logger.info("engine paused at startup (mining active), will load on resume")

    # Start scheduler listener (handles yield/resume going forward)
    try:
        if scheduler_client._stub:
            main_loop = asyncio.get_running_loop()
            listener = SchedulerListener(scheduler_client, engine, main_loop=main_loop)
            listener.start()
            logger.info("scheduler listener started")
    except Exception as e:
        logger.warning("could not start scheduler listener",
                       extra={"error": str(e)})

    # Create and run API server (pass model_manager for runtime model switching)
    app = create_app(
        engine,
        model_manager=model_manager,
        api_token=cfg.inference.api_token,
        max_tokens_limit=cfg.inference.max_tokens_limit,
    )
    if not cfg.inference.api_token:
        logger.warning("inference API auth is DISABLED — set INFERENCE_API_TOKEN "
                       "or firewall the inference port (the gateway is the auth layer)")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=cfg.inference.api_port,
        log_level=cfg.logging.level,
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        logger.info("shutting down...")
        if listener:
            listener.stop()
            scheduler_client.close()
        await engine.stop()
        logger.info("shutdown complete")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
