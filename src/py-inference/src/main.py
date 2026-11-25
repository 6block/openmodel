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
from .scheduler_client.token_store import GatewayTokenStore


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
    if cfg.pools:
        logger.info("multi-pool worker: %s",
                    ", ".join(f"{p.name}={p.inference.engine}@{p.inference.multi_gpu.device_ids}"
                              for p in cfg.pools))
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

    # Load model only if GPU is available; otherwise start paused. A multi-pool
    # worker resolves every pool's model (each may need an FOC download) and records
    # the local path on the pool before starting; the start() argument then only
    # names the primary pool's model.
    from src.model.quant_identity import check_quant_identity
    from src.model.model_kind import check_engine_fits
    if cfg.pools:
        for pc, pool in zip(cfg.pools, engine.pools):
            path = await model_manager.ensure_model(pc.inference.model)
            check_quant_identity(pc.inference.model, path)
            check_engine_fits(pc.inference.engine, pc.inference.model, path)
            for e in pool.engine._engines:
                e._model = path
        model_path = engine.current_model
    else:
        model_path = await model_manager.ensure_model(cfg.inference.model)
        check_quant_identity(cfg.inference.model, model_path)
        check_engine_fits(cfg.inference.engine, cfg.inference.model, model_path)
    # Switch targets already on disk are checked now, so a misplaced entry stops the
    # worker at startup with a clear reason instead of failing its first request.
    # Targets not yet downloaded are checked when a switch resolves them.
    from src.model.model_kind import check_switch_targets
    check_switch_targets(cfg, model_manager.present_path)
    if gpu_available:
        await engine.start(model_path)
        logger.info("engine started", extra={"model": model_path})
    else:
        await engine.start_paused(model_path)
        logger.info("engine paused at startup (mining active), will load on resume")

    # Dynamic gateway tokens (SP self-registration): the listener learns the
    # current per-worker token from the scheduler and the API server accepts it
    # alongside the static INFERENCE_API_TOKEN.
    token_store = GatewayTokenStore()

    # Start scheduler listener (handles yield/resume going forward)
    try:
        if scheduler_client._stub:
            main_loop = asyncio.get_running_loop()
            listener = SchedulerListener(scheduler_client, engine, main_loop=main_loop,
                                         token_store=token_store)
            listener.start()
            logger.info("scheduler listener started")
    except Exception as e:
        logger.warning("could not start scheduler listener",
                       extra={"error": str(e)})

    # Prefetch weights for every claimed model, in the background. Without this,
    # the download happens inside the first request-triggered switch — and that
    # request (admission probe: 120s; gateway: 300s) times out long before a
    # multi-minute download completes. Disk-only; never touches the engine.
    prefetch_task = None
    if cfg.gateway.supported_models:
        async def _prefetch_claimed():
            models = list(dict.fromkeys(cfg.gateway.supported_models))
            logger.info("prefetch: ensuring weights for %d claimed models", len(models))
            for m in models:
                try:
                    result = await model_manager.prefetch(m)
                    logger.info("prefetch %s: %s", m, result)
                except Exception as e:  # one bad entry must not kill the rest
                    logger.warning("prefetch %s failed: %s", m, e)
            logger.info("prefetch: done")
        prefetch_task = asyncio.create_task(_prefetch_claimed())

    # Create and run API server (pass model_manager for runtime model switching)
    app = create_app(
        engine,
        model_manager=model_manager,
        api_token=cfg.inference.api_token,
        max_tokens_limit=cfg.inference.max_tokens_limit,
        token_store=token_store,
    )
    if not cfg.inference.api_token:
        logger.warning("inference API auth has no static token — set INFERENCE_API_TOKEN "
                       "or rely on scheduler-delivered self-registration tokens; "
                       "otherwise firewall the inference port (the gateway is the auth layer)")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=cfg.inference.api_port,
        log_level=cfg.logging.level,
        # MUST exceed the gateway's idle-connection timeout. uvicorn's default is 5s,
        # while the gateway pools idle connections far longer — so any second request
        # arriving in that gap was written into a socket this side had already closed.
        # The gateway then waited for response headers that could never come and failed
        # the request only at its full request timeout (5 min on mainnet), with no trace
        # here: no access log, no engine activity, GPUs idle. Keep this comfortably
        # above the gateway value so the client is always the one to retire a connection.
        timeout_keep_alive=120,
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        logger.info("shutting down...")
        if prefetch_task:
            prefetch_task.cancel()
        if listener:
            listener.stop()
            scheduler_client.close()
        await engine.stop()
        logger.info("shutdown complete")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
