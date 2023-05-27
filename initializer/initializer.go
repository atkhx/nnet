package initializer

import "math"

const (
	reLuGain    = 1.4142135624 // √2
	tanhGain    = 1.6666666667 // 5/3
	sigmoidGain = 1.0
	linearGain  = 1.0
)

var (
	KaimingNormalReLU    = KaimingNormal{Gain: reLuGain}
	KaimingNormalTanh    = KaimingNormal{Gain: tanhGain}
	KaimingNormalSigmoid = KaimingNormal{Gain: sigmoidGain}
	KaimingNormalLinear  = KaimingNormal{Gain: linearGain}
)

type Initializer interface {
	GetNormK(fanIn int) float64
}

type InitWeightFixed struct {
	NormK float64
}

func (wi InitWeightFixed) GetNormK(fanIn int) float64 {
	return wi.NormK
}

type KaimingNormal struct {
	Gain float64
}

func (wi KaimingNormal) GetNormK(fanIn int) float64 {
	return wi.Gain / math.Sqrt(float64(fanIn))
}
