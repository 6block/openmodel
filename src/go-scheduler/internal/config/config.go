package config

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Mode      string          `yaml:"mode"` // "dev" or "prod"
	Lotus     LotusConfig     `yaml:"lotus"`
	Curio     CurioConfig     `yaml:"curio"`
	Scheduler SchedulerConfig `yaml:"scheduler"`
	GRPC      GRPCConfig      `yaml:"grpc"`
	Metrics   MetricsConfig   `yaml:"metrics"`
	Logging   LoggingConfig   `yaml:"logging"`
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
}

type CurioConfig struct {
	DSN                  string `yaml:"dsn"`
	ProofPollIntervalSec int    `yaml:"proof_poll_interval_sec"`
	ProofMaxWaitSec      int    `yaml:"proof_max_wait_sec"`
	LogPath              string `yaml:"log_path"` // Path to curio log file for WinningPoSt detection via log tailing
}

type WinningPostPolicy struct {
	Enabled       bool `yaml:"enabled"`
	ResumeDelaySec int  `yaml:"resume_delay_sec"`
	TimeoutSec    int  `yaml:"timeout_sec"`
}

type GRPCConfig struct {
	Port             int `yaml:"port"`
	MaxMessageSizeMB int `yaml:"max_message_size_mb"`
}

type MetricsConfig struct {
	Port    int  `yaml:"port"`
	Enabled bool `yaml:"enabled"`
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
	if cfg.Curio.ProofPollIntervalSec == 0 {
		cfg.Curio.ProofPollIntervalSec = 5
	}
	if cfg.Curio.ProofMaxWaitSec == 0 {
		cfg.Curio.ProofMaxWaitSec = 600
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
