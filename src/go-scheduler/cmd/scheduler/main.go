package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"openmodel/go-scheduler/internal/config"
	"openmodel/go-scheduler/internal/curio"
	"openmodel/go-scheduler/internal/grpcapi"
	"openmodel/go-scheduler/internal/health"
	"openmodel/go-scheduler/internal/lotus"
	"openmodel/go-scheduler/internal/scheduler"
	"openmodel/go-scheduler/internal/spregister"
	"openmodel/go-scheduler/internal/tlsfront"
	pb "openmodel/go-scheduler/proto/sidecar"
)

// parseMinerID extracts the numeric miner ID from a t0/f0-prefixed Filecoin
// miner address (e.g. "t0182063" → 182063). Returns 0 for malformed input.
func parseMinerID(addr string) int64 {
	var id int64
	if len(addr) > 2 && (addr[:2] == "t0" || addr[:2] == "f0") {
		fmt.Sscanf(addr[2:], "%d", &id)
	}
	return id
}

func main() {
	configPath := flag.String("config", "/etc/sidecar/sidecar-prod-test.yaml", "path to config file")
	registerOnly := flag.Bool("register-only", false, "run gateway SP self-registration once, write the token file, and exit (no scheduling)")
	flag.Parse()

	// Setup logger
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	// Load config
	cfg, err := config.Load(*configPath)
	if err != nil {
		logger.Error("failed to load config", "error", err)
		os.Exit(1)
	}
	logger.Info("config loaded", "mode", cfg.Mode)

	// Create Lotus client based on mode
	var lotusClient lotus.Client
	if cfg.Mode == "dev" {
		logger.Info("using mock lotus client (dev mode)")
		lotusClient = lotus.NewMockClient()
	} else {
		rpcClient := lotus.NewRPCClient(
			cfg.Lotus.APIURL,
			cfg.Lotus.APIToken,
			cfg.Lotus.MinerAddress,
			logger,
		)
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.Lotus.ConnectTimeoutSec)*time.Second)
		if err := rpcClient.Connect(ctx, time.Duration(cfg.Lotus.ConnectTimeoutSec)*time.Second); err != nil {
			cancel()
			logger.Error("failed to connect to lotus", "error", err)
			os.Exit(1)
		}
		cancel()
		lotusClient = rpcClient
	}
	defer lotusClient.Close()

	// SP self-registration with the gateway (miner-signed). Needs the REAL Lotus
	// client (wallet signing); dev/mock mode cannot register.
	// Built whenever a gateway is configured — NOT only when auto_register is on.
	// With auto_register: false the worker may still hold a token from an earlier
	// registration; skipping the Registrar would stop /ready from accepting it and
	// the gateway's polls would 401 the worker offline. Registration itself is still
	// gated on auto_register below.
	var registrar *spregister.Registrar
	if cfg.Gateway.URL != "" {
		rpcClient, ok := lotusClient.(*lotus.RPCClient)
		if !ok {
			// Only fatal when registration is actually going to run: a dev/mock
			// worker that merely carries a gateway block in its config must still
			// boot (it just cannot sign).
			if cfg.Gateway.AutoRegister || *registerOnly {
				logger.Error("gateway self-registration requires a real Lotus client (mode != dev)")
				os.Exit(1)
			}
			logger.Warn("gateway configured but self-registration needs a real Lotus client; skipping registrar")
		} else {
			registrar = spregister.New(spregister.Config{
				GatewayURL:         cfg.Gateway.URL,
				GatewayID:          cfg.Gateway.GatewayID,
				WorkerID:           cfg.Gateway.WorkerID,
				MinerID:            cfg.Lotus.MinerAddress,
				PayoutAddress:      cfg.Gateway.PayoutAddress,
				PublicEndpoint:     cfg.Gateway.PublicEndpoint,
				PublicSchedulerURL: cfg.Gateway.PublicSchedulerURL,
				GPUCount:           cfg.Gateway.GPUCount,
				SupportedModels:    cfg.Gateway.SupportedModels,
				TLSCAPath:          cfg.Gateway.TLSCAPath,
				TLSSystemCA:        cfg.Gateway.TLSSystemCA,
				TokenPath:          cfg.Gateway.TokenPath,
				AutoRegister:       cfg.Gateway.AutoRegister,
			}, rpcClient, logger)
			// Certificate-at-registration + built-in mTLS front. The key stays in
			// cert_dir forever; certificates arrive with registration and rotate
			// via the renew loop — no reload, no external terminator.
			if cfg.Gateway.MTLSFront.Enabled {
				cs, err := spregister.NewCertStore(cfg.Gateway.MTLSFront.CertDir, cfg.Gateway.WorkerID)
				if err != nil {
					logger.Error("mTLS cert store init failed", "error", err)
					os.Exit(1)
				}
				registrar.SetCertStore(cs)
				front, err := tlsfront.New(cs, logger, map[int]string{
					cfg.Gateway.MTLSFront.InferencePort: cfg.Gateway.MTLSFront.InferenceBackend,
					cfg.Gateway.MTLSFront.SchedulerPort: cfg.Gateway.MTLSFront.SchedulerBackend,
				})
				if err != nil {
					logger.Error("mTLS front init failed", "error", err)
					os.Exit(1)
				}
				front.Start()
			}
		}
	}

	// One-shot manual mode: register, persist the token file, exit. The token
	// value itself is deliberately NOT printed (secrets don't go to terminals) —
	// read it from token_path.
	if *registerOnly {
		if registrar == nil {
			logger.Error("register-only: gateway.url is not configured")
			os.Exit(1)
		}
		regCtx, regCancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer regCancel()
		res, err := registrar.RegisterOnce(regCtx)
		if err != nil {
			logger.Error("registration failed", "error", err)
			os.Exit(1)
		}
		logger.Info("registration complete",
			"worker_id", res.WorkerID, "miner", res.MinerID, "rotated", res.Rotated,
			"token_path", cfg.Gateway.TokenPath)
		os.Exit(0)
	}

	// Create scheduler
	policy := scheduler.YieldPolicy{
		WindowPost:           cfg.Scheduler.WindowPost,
		WinningPost:          cfg.Scheduler.WinningPost,
		FailSafeOnDisconnect: cfg.Scheduler.FailSafeOnDisconnect,
		ChainStaleAfter:      time.Duration(cfg.Scheduler.ChainStaleAfterSec) * time.Second,
	}
	sched := scheduler.New(lotusClient, policy, logger)

	// Setup proof completion monitor (Curio DB or mock)
	if cfg.Scheduler.WindowPost.ProofDetectionEnabled {
		proofWaitCfg := curio.WaitConfig{
			PollInterval: time.Duration(cfg.Curio.ProofPollIntervalSec) * time.Second,
			MaxWait:      time.Duration(cfg.Curio.ProofMaxWaitSec) * time.Second,
		}

		minerID := parseMinerID(cfg.Lotus.MinerAddress)

		if cfg.Mode == "dev" {
			mockMonitor := curio.NewMockProofMonitor(30*time.Second, logger)
			sched.SetProofMonitor(mockMonitor, minerID, proofWaitCfg)
			logger.Info("using mock proof monitor (dev mode)", "proof_delay", "30s")
		} else if cfg.Curio.DSN != "" {
			dbClient, err := curio.NewDBClient(cfg.Curio.DSN, logger)
			if err != nil {
				logger.Error("failed to connect to curio DB, proof detection disabled", "error", err)
			} else {
				sched.SetProofMonitor(dbClient, minerID, proofWaitCfg)
				defer dbClient.Close()
			}
		} else {
			// Loud, not silent: an empty DSN usually means the config used keys
			// the struct does not have (db_host/db_port…) — YAML drops unknown
			// keys without complaint, and this exact combination kept WinningPoSt
			// detection off on a production worker for two weeks unnoticed.
			logger.Warn("curio.dsn is empty — proof detection AND WinningPoSt monitoring are DISABLED; " +
				"if you meant to configure Curio, use a single dsn string (split db_host/db_port keys are not read)")
		}
	}

	// Setup Curio log watcher for fast WinningPoSt detection
	if cfg.Curio.LogPath != "" {
		watcher := curio.NewLogWatcher(cfg.Curio.LogPath, logger)
		sched.SetCurioLogWatcher(watcher)
		logger.Info("Curio log watcher configured", "path", cfg.Curio.LogPath)
	}

	// Create gRPC server
	grpcHandler := grpcapi.NewHandler(sched, logger)
	if registrar != nil {
		// Inference learns the (rotating) gateway token via its 3s poll.
		grpcHandler.SetGatewayTokenProvider(registrar.CurrentToken)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterSchedulerServiceServer(grpcServer, grpcHandler)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.GRPC.Port))
	if err != nil {
		logger.Error("failed to listen", "port", cfg.GRPC.Port, "error", err)
		os.Exit(1)
	}

	// Start scheduler
	ctx, cancel := context.WithCancel(context.Background())

	// Kick off gateway self-registration in the background — it retries on its
	// own and never blocks mining protection.
	if registrar != nil && cfg.Gateway.AutoRegister {
		registrar.RunBackground(ctx)
		registrar.RunRenewLoop(ctx)
	}

	// Start health/metrics server
	var healthServer *health.Server
	if cfg.Metrics.Enabled {
		healthServer = health.NewServer(cfg.Metrics.Port, sched, logger, ctx, cfg.Metrics.AuthToken)
		if registrar != nil {
			// /ready must accept the freshly-rotated token the gateway polls with.
			healthServer.SetDynamicToken(registrar.CurrentToken)
			// GET /registration lets the SP operator check registration state without
			// grepping logs or opening the token file.
			healthServer.SetRegistrationStatus(func() any { return registrar.CurrentStatus() })
			healthServer.SetReverify(registrar.Reverify)
			// Each gateway poll is the only proof the registration is still honoured;
			// the watcher turns its ABSENCE into a diagnosis (network vs deregistered).
			healthServer.SetGatewayPollObserver(registrar.NoteGatewayPoll)
			registrar.WatchLink(ctx)
		}
		go healthServer.Start()
		logger.Info("metrics server started", "port", cfg.Metrics.Port)

		// Wire scheduler state changes to Prometheus yield counter
		sched.SetOnStateChange(func(state pb.GpuState, reason pb.YieldReason) {
			// Only count actual yields, not resumes back to AVAILABLE
			if state != pb.GpuState_GPU_STATE_AVAILABLE && state != pb.GpuState_GPU_STATE_UNKNOWN {
				health.IncYieldEvent(reason.String())
			}
		})
	}
	defer cancel()

	go sched.Run(ctx, time.Duration(cfg.Lotus.PollIntervalSec)*time.Second)
	logger.Info("scheduler started", "poll_interval_sec", cfg.Lotus.PollIntervalSec)

	// Start gRPC server
	go func() {
		logger.Info("grpc server started", "port", cfg.GRPC.Port)
		if err := grpcServer.Serve(lis); err != nil {
			logger.Error("grpc server error", "error", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	logger.Info("shutting down", "signal", sig)

	cancel()
	grpcServer.GracefulStop()
	if healthServer != nil {
		healthServer.Stop()
	}
	logger.Info("shutdown complete")
}
