package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewEmbed(
	featuresCount int,
	alphabetSize int,
	gain float64,
) *Embed {
	return &Embed{
		alphabetSize:  alphabetSize,
		featuresCount: featuresCount,
		gain:          gain,
	}
}

type Embed struct {
	featuresCount int
	alphabetSize  int

	gain float64

	iSize int
	bSize int

	inputIdxByValue []int

	// internal buffers
	Weights num.Float64s // (storable)
	wGrads  num.Float64s

	// buffers from the previous layer
	inputs num.Float64s
	//iGrads num.Float64s

	// buffers to the next layer
	output num.Float64s
	oGrads num.Float64s
}

func (l *Embed) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	weights := make(num.Float64s, l.alphabetSize*l.featuresCount)
	weights.RandNorm()

	l.Weights = weights
	l.wGrads = make(num.Float64s, l.alphabetSize*l.featuresCount)

	l.inputs = inputs

	l.output = make(num.Float64s, l.featuresCount*l.iSize*l.bSize)
	l.oGrads = make(num.Float64s, l.featuresCount*l.iSize*l.bSize)

	l.inputIdxByValue = make([]int, len(inputs))

	return l.output, l.oGrads
}

func (l *Embed) Forward() {
	for i, v := range l.inputs {
		l.inputIdxByValue[i] = int(v)
	}

	for b := 0; b < l.bSize; b++ {
		inputIdxByValue := l.inputIdxByValue[b*l.iSize : (b+1)*l.iSize]

		output := l.output[b*l.featuresCount*l.iSize : (b+1)*l.featuresCount*l.iSize]

		for i, idx := range inputIdxByValue {
			features := l.Weights[idx*l.featuresCount : (idx+1)*l.featuresCount]
			output := output[i*l.featuresCount : (i+1)*l.featuresCount]

			copy(output, features)
		}
	}
}

func (l *Embed) Backward() {
	for b := 0; b < l.bSize; b++ {
		inputs := l.inputIdxByValue[b*l.iSize : (b+1)*l.iSize]
		oGrads := l.oGrads[b*l.featuresCount*l.iSize : (b+1)*l.featuresCount*l.iSize]

		for i, idx := range inputs {
			wGrads := l.wGrads[idx*l.featuresCount : (idx+1)*l.featuresCount]
			wGrads.Add(oGrads[i*l.featuresCount : (i+1)*l.featuresCount])
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
