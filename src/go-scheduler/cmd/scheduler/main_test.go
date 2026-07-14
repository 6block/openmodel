package main

import "testing"

func TestParseMinerID(t *testing.T) {
	cases := map[string]int64{
		"t0182063": 182063, // testnet
		"f0185520": 185520, // mainnet
		"":         0,
		"x":        0,
		"t1abc":    0, // wrong prefix (t1, not t0)
		"0182063":  0, // no prefix
		"t0abc":    0, // non-numeric body → 0
	}
	for in, want := range cases {
		if got := parseMinerID(in); got != want {
			t.Errorf("parseMinerID(%q) = %d, want %d", in, got, want)
		}
	}
}
