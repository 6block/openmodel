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

	// Create scheduler
	policy := scheduler.YieldPolicy{
		WindowPost:  cfg.Scheduler.WindowPost,
		WinningPost: cfg.Scheduler.WinningPost,
		FailSafeOnDisconnect: cfg.Scheduler.FailSafeOnDisconnect,
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
	grpcServer := grpc.NewServer()
	pb.RegisterSchedulerServiceServer(grpcServer, grpcHandler)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.GRPC.Port))
	if err != nil {
		logger.Error("failed to listen", "port", cfg.GRPC.Port, "error", err)
		os.Exit(1)
	}

	// Start scheduler
	ctx, cancel := context.WithCancel(context.Background())

	// Start health/metrics server
	var healthServer *health.Server
	if cfg.Metrics.Enabled {
		healthServer = health.NewServer(cfg.Metrics.Port, sched, logger, ctx, cfg.Metrics.AuthToken)
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
