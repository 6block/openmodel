package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInterpolateEnvVars(t *testing.T) {
	t.Setenv("FOO_VAR", "hello")
	t.Setenv("BAR_VAR", "world")

	if got := interpolateEnvVars("${FOO_VAR}-${BAR_VAR}"); got != "hello-world" {
		t.Errorf("substitution = %q, want %q", got, "hello-world")
	}
	if got := interpolateEnvVars("x=${DEFINITELY_UNSET_VAR_XYZ}"); got != "x=" {
		t.Errorf("unset var = %q, want %q", got, "x=")
	}
	if got := interpolateEnvVars("plain text"); got != "plain text" {
		t.Errorf("no-ref = %q, want unchanged", got)
	}
	// unterminated ${ → loop breaks, left as-is
	if got := interpolateEnvVars("a ${UNCLOSED here"); got != "a ${UNCLOSED here" {
		t.Errorf("unterminated = %q, want unchanged", got)
	}
}

func TestLoadReadError(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "nonexistent.yaml")); err == nil {
		t.Fatal("expected an error for a nonexistent config file")
	}
}

func TestLoadParseError(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(p, []byte("mode: [unterminated flow seq"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(p); err == nil {
		t.Fatal("expected a parse error for invalid YAML")
	}
}

func TestLoadInterpolatesAndAppliesDefaults(t *testing.T) {
	t.Setenv("TEST_MODE_VAR", "prod")
	p := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(p, []byte("mode: \"${TEST_MODE_VAR}\"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Mode != "prod" {
		t.Errorf("interpolated mode = %q, want %q", cfg.Mode, "prod")
	}
	// a default was applied (proves applyDefaults ran after parse)
	if cfg.Lotus.PollIntervalSec != 15 {
		t.Errorf("default PollIntervalSec = %d, want 15", cfg.Lotus.PollIntervalSec)
	}
}
