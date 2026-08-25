package config

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Mode      string           `yaml:"mode"` // "dev" or "prod"
	Lotus     LotusConfig      `yaml:"lotus"`
	Curio     CurioConfig      `yaml:"curio"`
	Scheduler SchedulerConfig  `yaml:"scheduler"`
	GRPC      GRPCConfig       `yaml:"grpc"`
	Metrics   MetricsConfig    `yaml:"metrics"`
	Gateway   GatewayRegConfig `yaml:"gateway"`
	Logging   LoggingConfig    `yaml:"logging"`
}

// GatewayRegConfig configures SP self-registration with an OpenModel gateway.
// When auto_register is set, the scheduler proves miner control by signing the
// gateway's challenge with the miner's worker/owner key from the local Lotus
// wallet, receives the per-worker auth token, and hands it to the inference
// service over gRPC — no operator action needed on either side.
type GatewayRegConfig struct {
	URL string `yaml:"url"` // gateway base URL, e.g. "http://gateway-host:3000"
	// GatewayID must equal the gateway's sp_registration.gateway_id — the signed
	// message is domain-bound to it so it cannot be replayed against another gateway.
	GatewayID string `yaml:"gateway_id"`
	WorkerID  string `yaml:"worker_id"` // default: "sp-" + miner address
	// PayoutAddress is the EVM address this SP's earnings settle to. It is part
	// of the miner-signed message, so nobody (gateway included) can redirect it.
	PayoutAddress string `yaml:"payout_address"`
	// PublicEndpoint / PublicSchedulerURL are this worker's py-inference and
	// scheduler URLs as reachable FROM the gateway (public IP or tunnel).
	PublicEndpoint     string   `yaml:"public_endpoint"`
	PublicSchedulerURL string   `yaml:"public_scheduler_url"`
	GPUCount           int      `yaml:"gpu_count"`
	SupportedModels    []string `yaml:"supported_models"`
	// TLSCAPath is the trust anchor for an https gateway.url — the registration
	// CA certificate, distributed with onboarding materials. Once a registration
	// succeeds the cert store holds the same CA, so this matters mainly for the
	// very first (cold) registration.
	TLSCAPath string `yaml:"tls_ca_path"`
	// TLSSystemCA switches the https gateway.url to PUBLIC trust: the system
	// certificate store validates the chain and the URL's hostname is the
	// expected identity — exactly like any ordinary HTTPS client. For gateways
	// served on a real domain behind a publicly-trusted certificate; no CA file
	// to distribute. Mutually exclusive with tls_ca_path (private-CA mode,
	// identity = gateway_id): setting both is a configuration error.
	TLSSystemCA  bool `yaml:"tls_system_ca"`
	AutoRegister bool `yaml:"auto_register"`
	// TokenPath persists the issued token (0600) so restarts stay authenticated;
	// mount a small volume there in Docker. Default /data/gateway-token.json.
	TokenPath string `yaml:"token_path"`
	// MTLSFront enables the built-in mTLS front: registration submits a CSR (the
	// private key NEVER leaves this machine), the gateway signs a short-lived
	// worker certificate, and this scheduler terminates mTLS itself on the two
	// ports below — no nginx, no reload (certificates hot-swap per handshake).
	// Renewal runs automatically at half the certificate lifetime; a banned or
	// deregistered worker is refused renewal and ages out of the mTLS layer.
	MTLSFront MTLSFrontConfig `yaml:"mtls_front"`
}

// MTLSFrontConfig — see GatewayRegConfig.MTLSFront.
type MTLSFrontConfig struct {
	Enabled bool `yaml:"enabled"`
	// Listen ports for the TLS side. Point public_endpoint / public_scheduler_url
	// at these (https://...) once enabled.
	InferencePort int `yaml:"inference_port"` // default 8443 → proxies 127.0.0.1:8000
	SchedulerPort int `yaml:"scheduler_port"` // default 9443 → proxies 127.0.0.1:9090
	// Backends being fronted (plaintext, local).
	InferenceBackend string `yaml:"inference_backend"` // default http://127.0.0.1:8000
	SchedulerBackend string `yaml:"scheduler_backend"` // default http://127.0.0.1:9090
	// CertDir persists worker.key / worker.crt / ca.crt (0600 key). Mount the
	// same volume as token_path. Default /data/mtls.
	CertDir string `yaml:"cert_dir"`
}

type LotusConfig struct {
	APIURL            string `yaml:"api_url"`
	APIToken          string `yaml:"api_token"`
	MinerAddress      string `yaml:"miner_address"`
	PollIntervalSec   int    `yaml:"poll_interval_sec"`
	ConnectTimeoutSec int    `yaml:"connect_timeout_sec"`
}

type SchedulerConfig struct {
	WindowPost           WindowPostPolicy  `yaml:"window_post"`
	WinningPost          WinningPostPolicy `yaml:"winning_post"`
	FailSafeOnDisconnect bool              `yaml:"fail_safe_on_disconnect"`
}

type WindowPostPolicy struct {
	GracefulYieldThresholdSec int  `yaml:"graceful_yield_threshold_sec"`
	HardStopThresholdSec      int  `yaml:"hard_stop_threshold_sec"`
	ResumeDelayAfterCloseSec  int  `yaml:"resume_delay_after_close_sec"`
	YieldDuringFaultCutoff    bool `yaml:"yield_during_fault_cutoff"`
	ProofDetectionEnabled     bool `yaml:"proof_detection_enabled"`
	// How often (in chain epochs) to refresh the full per-deadline sector cache.
	// Smaller = newly-sealed sectors are detected sooner (so we yield for their
	// WindowPoSt) at the cost of more Lotus queries. Default 360 (~3h).
	SectorCacheRefreshEpochs int `yaml:"sector_cache_refresh_epochs"`
}

type CurioConfig struct {
	DSN                  string `yaml:"dsn"`
	ProofPollIntervalSec int    `yaml:"proof_poll_interval_sec"`
	ProofMaxWaitSec      int    `yaml:"proof_max_wait_sec"`
	LogPath              string `yaml:"log_path"` // Path to curio log file for WinningPoSt detection via log tailing
}

type WinningPostPolicy struct {
	Enabled        bool `yaml:"enabled"`
	ResumeDelaySec int  `yaml:"resume_delay_sec"`
	TimeoutSec     int  `yaml:"timeout_sec"`
}

type GRPCConfig struct {
	Port             int `yaml:"port"`
	MaxMessageSizeMB int `yaml:"max_message_size_mb"`
}

type MetricsConfig struct {
	Port    int  `yaml:"port"`
	Enabled bool `yaml:"enabled"`
	// AuthToken, if set, requires `Authorization: Bearer <token>` on /ready and the
	// /debug/* endpoints (served on this port). The gateway sends the per-worker
	// token on its /ready polls; set this to the same value the worker is registered
	// with. Empty = no auth (only safe on a trusted LAN). /health and /metrics stay
	// open (local container healthcheck / internal Prometheus scrape — firewall them).
	// Use `auth_token: ${SCHEDULER_AUTH_TOKEN}` to source it from the environment.
	AuthToken string `yaml:"auth_token"`
}

type LoggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

// Load reads and parses the config file, interpolating environment variables.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	// Interpolate ${ENV_VAR} references
	content := interpolateEnvVars(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(content), &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	applyDefaults(&cfg)
	if cfg.Gateway.TLSSystemCA && cfg.Gateway.TLSCAPath != "" {
		return nil, fmt.Errorf("gateway.tls_system_ca and gateway.tls_ca_path are mutually exclusive: " +
			"system trust validates the URL hostname via public CAs, tls_ca_path pins the private " +
			"registration CA (identity = gateway_id) — configure exactly one")
	}
	return &cfg, nil
}

func interpolateEnvVars(content string) string {
	result := content
	for {
		start := strings.Index(result, "${")
		if start == -1 {
			break
		}
		end := strings.Index(result[start:], "}")
		if end == -1 {
			break
		}
		end += start
		envVar := result[start+2 : end]
		envVal := os.Getenv(envVar)
		result = result[:start] + envVal + result[end+1:]
	}
	return result
}

func applyDefaults(cfg *Config) {
	if cfg.Mode == "" {
		cfg.Mode = "dev"
	}
	if cfg.Lotus.PollIntervalSec == 0 {
		cfg.Lotus.PollIntervalSec = 15
	}
	if cfg.Lotus.ConnectTimeoutSec == 0 {
		cfg.Lotus.ConnectTimeoutSec = 10
	}
	if cfg.Scheduler.WindowPost.GracefulYieldThresholdSec == 0 {
		cfg.Scheduler.WindowPost.GracefulYieldThresholdSec = 300
	}
	if cfg.Scheduler.WindowPost.HardStopThresholdSec == 0 {
		cfg.Scheduler.WindowPost.HardStopThresholdSec = 120
	}
	if cfg.Scheduler.WindowPost.ResumeDelayAfterCloseSec == 0 {
		cfg.Scheduler.WindowPost.ResumeDelayAfterCloseSec = 60
	}
	if cfg.Scheduler.WindowPost.SectorCacheRefreshEpochs == 0 {
		// ~3h. Was a hardcoded 2880 (~24h), too stale for actively-sealing miners.
		cfg.Scheduler.WindowPost.SectorCacheRefreshEpochs = 360
	}
	if cfg.Curio.ProofPollIntervalSec == 0 {
		cfg.Curio.ProofPollIntervalSec = 5
	}
	if cfg.Curio.ProofMaxWaitSec == 0 {
		cfg.Curio.ProofMaxWaitSec = 600
	}
	if cfg.Gateway.WorkerID == "" {
		cfg.Gateway.WorkerID = "sp-" + cfg.Lotus.MinerAddress
	}
	if cfg.Gateway.TokenPath == "" {
		cfg.Gateway.TokenPath = "/data/gateway-token.json"
	}
	if cfg.Gateway.MTLSFront.InferencePort == 0 {
		cfg.Gateway.MTLSFront.InferencePort = 8443
	}
	if cfg.Gateway.MTLSFront.SchedulerPort == 0 {
		cfg.Gateway.MTLSFront.SchedulerPort = 9443
	}
	if cfg.Gateway.MTLSFront.InferenceBackend == "" {
		cfg.Gateway.MTLSFront.InferenceBackend = "http://127.0.0.1:8000"
	}
	if cfg.Gateway.MTLSFront.SchedulerBackend == "" {
		cfg.Gateway.MTLSFront.SchedulerBackend = "http://127.0.0.1:9090"
	}
	if cfg.Gateway.MTLSFront.CertDir == "" {
		cfg.Gateway.MTLSFront.CertDir = "/data/mtls"
	}
	if cfg.Gateway.GPUCount == 0 {
		cfg.Gateway.GPUCount = 1
	}
	if cfg.Scheduler.WinningPost.ResumeDelaySec == 0 {
		cfg.Scheduler.WinningPost.ResumeDelaySec = 5
	}
	if cfg.Scheduler.WinningPost.TimeoutSec == 0 {
		cfg.Scheduler.WinningPost.TimeoutSec = 25
	}
	if cfg.GRPC.Port == 0 {
		cfg.GRPC.Port = 50051
	}
	if cfg.GRPC.MaxMessageSizeMB == 0 {
		cfg.GRPC.MaxMessageSizeMB = 4
	}
	if cfg.Metrics.Port == 0 {
		cfg.Metrics.Port = 9090
	}
	if cfg.Logging.Level == "" {
		cfg.Logging.Level = "info"
	}
	if cfg.Logging.Format == "" {
		cfg.Logging.Format = "json"
	}
}
