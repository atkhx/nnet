package layer

import "math"

const (
	reLuGain    = 1.4142135624
	tanhGain    = 1.6666666667
	sigmoidGain = 1.0
	linearGain  = 1.0
)

var (
	WithReLuGain    = InitWeightLinearWithGain{Gain: reLuGain}
	WithTanhGain    = InitWeightLinearWithGain{Gain: tanhGain}
	WithSigmoidGain = InitWeightLinearWithGain{Gain: sigmoidGain}
	WithLinearGain  = InitWeightLinearWithGain{Gain: linearGain}
)

type InitWeights interface {
	GetNormK(fanIn int) float64
}

type InitWeightFixed struct {
	NormK float64
}

func (wi InitWeightFixed) GetNormK(fanIn int) float64 {
	return wi.NormK
}

type InitWeightLinearWithGain struct {
	Gain float64
}

func (wi InitWeightLinearWithGain) GetNormK(fanIn int) float64 {
	return wi.Gain / math.Pow(float64(fanIn), 0.5)
}
