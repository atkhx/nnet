package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewEmbed(
	featuresCount int,
	contextLength int,
	gain float64,
) *Embed {
	return &Embed{
		featuresCount: featuresCount,
		contextLength: contextLength,

		gain: gain,
	}
}

type Embed struct {
	featuresCount int
	contextLength int
	alphabetSize  int

	gain float64

	iSize int
	bSize int

	// internal buffers
	Weights num.Float64s // (storable)
	wGrads  num.Float64s

	// buffers from the previous layer
	inputs num.Float64s
	iGrads num.Float64s

	// buffers to the next layer
	output num.Float64s
	oGrads num.Float64s
}

func (l *Embed) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	l.alphabetSize = l.iSize / l.contextLength

	weightK := 1.0
	if l.gain > 0 {
		fanIn := l.iSize * l.bSize
		weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	}

	weights := make(num.Float64s, l.alphabetSize*l.featuresCount)
	weights.RandNormWeighted(weightK)

	l.Weights = weights
	l.wGrads = make(num.Float64s, l.alphabetSize*l.featuresCount)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, l.featuresCount*l.contextLength*l.bSize)
	l.oGrads = make(num.Float64s, l.featuresCount*l.contextLength*l.bSize)

	return l.output, l.oGrads
}

func (l *Embed) Forward() {
	for b := 0; b < l.bSize*l.contextLength; b++ {
		inputs := l.inputs[b*l.alphabetSize : (b+1)*l.alphabetSize]
		output := l.output[b*l.featuresCount : (b+1)*l.featuresCount]

		for o := 0; o < l.featuresCount; o++ {
			output[o] = num.Dot(inputs, l.Weights[o*l.alphabetSize:(o+1)*l.alphabetSize])
		}
	}
}

func (l *Embed) Backward() {
	for b := 0; b < l.bSize*l.contextLength; b++ {
		inputs := l.inputs[b*l.alphabetSize : (b+1)*l.alphabetSize]
		iGrads := l.iGrads[b*l.alphabetSize : (b+1)*l.alphabetSize]

		for i, delta := range l.oGrads[b*l.featuresCount : (b+1)*l.featuresCount] {
			iGrads.AddWeighted(l.Weights[i*l.alphabetSize:(i+1)*l.alphabetSize], delta)
			l.wGrads[i*l.alphabetSize:(i+1)*l.alphabetSize].AddWeighted(inputs, delta)
		}
	}
}

func (l *Embed) ResetGrads() {
	l.oGrads.Fill(0)
	l.wGrads.Fill(0)
}

func (l *Embed) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.wGrads}}
}
