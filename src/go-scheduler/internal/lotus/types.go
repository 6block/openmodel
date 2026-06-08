package lotus

// DeadlineInfo mirrors dline.Info from go-state-types.
// We define our own struct to avoid pulling the full Filecoin dependency tree.
type DeadlineInfo struct {
	CurrentEpoch           int64  `json:"CurrentEpoch"`
	PeriodStart            int64  `json:"PeriodStart"`
	Index                  uint64 `json:"Index"`
	Open                   int64  `json:"Open"`
	Close                  int64  `json:"Close"`
	Challenge              int64  `json:"Challenge"`
	FaultCutoff            int64  `json:"FaultCutoff"`
	WPoStPeriodDeadlines   uint64 `json:"WPoStPeriodDeadlines"`
	WPoStProvingPeriod     int64  `json:"WPoStProvingPeriod"`
	WPoStChallengeWindow   int64  `json:"WPoStChallengeWindow"`
	WPoStChallengeLookback int64  `json:"WPoStChallengeLookback"`
	FaultDeclarationCutoff int64  `json:"FaultDeclarationCutoff"`
}

// EpochDuration is the duration of one Filecoin epoch in seconds.
const EpochDuration = 30

// SecondsUntilOpen returns wall-clock seconds until the deadline window opens.
func (d *DeadlineInfo) SecondsUntilOpen() int64 {
	epochs := d.Open - d.CurrentEpoch
	if epochs < 0 {
		return 0
	}
	return epochs * EpochDuration
}

// SecondsUntilClose returns wall-clock seconds until the deadline window closes.
func (d *DeadlineInfo) SecondsUntilClose() int64 {
	epochs := d.Close - d.CurrentEpoch
	if epochs < 0 {
		return 0
	}
	return epochs * EpochDuration
}

// IsOpen returns true if the deadline window is currently open.
func (d *DeadlineInfo) IsOpen() bool {
	return d.CurrentEpoch >= d.Open && d.CurrentEpoch < d.Close
}

// DeadlineSectors contains sector count info for a specific proving deadline.
type DeadlineSectors struct {
	Deadline   uint64 `json:"Deadline"`
	Partitions int    `json:"Partitions"`
	Sectors    int    `json:"Sectors"`
	Faults     int    `json:"Faults"`
}

// HasSectors returns true if this deadline has sectors that need proving.
func (ds *DeadlineSectors) HasSectors() bool {
	return ds.Sectors > 0
}

// MinerBaseInfo contains the result of MinerGetBaseInfo, used to check
// WinningPoSt eligibility for a given epoch.
type MinerBaseInfo struct {
	// HasMinPower indicates the miner has minimum power to participate in consensus.
	HasMinPower bool `json:"HasMinPower"`
	// EligibleForMining indicates the miner is eligible to mine a block at this epoch.
	EligibleForMining bool `json:"EligibleForMining"`
}

// TipsetCID is a CID reference in a tipset key.
type TipsetCID struct {
	Root string `json:"/"`
}

// ChainHead represents a minimal chain head notification.
type ChainHead struct {
	Height int64       `json:"Height"`
	Cids   []TipsetCID `json:"Cids,omitempty"` // Tipset key for MinerGetBaseInfo
}
