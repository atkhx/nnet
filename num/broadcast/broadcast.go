package broadcast

import "github.com/atkhx/nnet/num"

func NewConfig(aDims, bDims num.Dims) Config {
	steps := MakeSteps(aDims, bDims)
	oDims := MakeOutDims(aDims, bDims)

	return Config{
		BCSteps: steps,
		OutDims: oDims,
		Offsets: MakeOffsets(steps, oDims),
	}
}

type Config struct {
	BCSteps Steps
	OutDims num.Dims
	Offsets []Offsets
}

func (cfg *Config) Broadcast(fn func(aOffset, bOffset, oOffset int)) {
	for _, offset := range cfg.Offsets {
		fn(offset.AOffset, offset.BOffset, offset.OOffset)
	}
}

func MakeOutDims(aDims, bDims num.Dims) num.Dims {
	maxVal := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}

	return num.NewDims(
		maxVal(aDims.W, bDims.W),
		maxVal(aDims.H, bDims.H),
		maxVal(aDims.D, bDims.D),
	)
}
