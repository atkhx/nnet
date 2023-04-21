package layer

import (
	"fmt"

	"github.com/atkhx/nnet/num"
)

func NewEmbedPos(
	featuresCount int,
	alphabetSize int,
	gain float64,
) *EmbedPos {
	return &EmbedPos{
		alphabetSize:  alphabetSize,
		featuresCount: featuresCount,
		gain:          gain,
	}
}

type EmbedPos struct {
	featuresCount int
	alphabetSize  int

	gain float64

	iSize int
	bSize int

	pos []int

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

func (l *EmbedPos) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	fmt.Println("l.iSize", l.iSize)
	fmt.Println("l.bSize", l.bSize)

	//weightK := 1.0
	//if l.gain > 0 {
	//	fanIn := l.iSize * l.bSize
	//	weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	//}

	weights := make(num.Float64s, l.alphabetSize*l.featuresCount)
	weights.RandNorm()
	//weights.RandNormWeighted(weightK)

	l.Weights = weights
	l.wGrads = make(num.Float64s, l.alphabetSize*l.featuresCount)

	l.inputs = inputs
	//l.iGrads = iGrads

	l.output = make(num.Float64s, l.featuresCount*l.iSize*l.bSize)
	l.oGrads = make(num.Float64s, l.featuresCount*l.iSize*l.bSize)

	l.pos = make([]int, len(inputs))

	return l.output, l.oGrads
}

func (l *EmbedPos) Forward() {
	for i, v := range l.inputs {
		l.pos[i] = int(v)
	}

	for b := 0; b < l.bSize; b++ {
		inputs := l.pos[b*l.iSize : (b+1)*l.iSize]
		output := l.output[b*l.featuresCount*l.iSize : (b+1)*l.featuresCount*l.iSize]

		for i, pos := range inputs {
			//feature := l.Weights[pos*l.featuresCount : (pos+1)*l.featuresCount]
			copy(output[i*l.featuresCount:(i+1)*l.featuresCount], l.Weights[pos*l.featuresCount:(pos+1)*l.featuresCount])
		}
	}
}

func (l *EmbedPos) Backward() {
	for b := 0; b < l.bSize; b++ {
		inputs := l.pos[b*l.iSize : (b+1)*l.iSize]
		oGrads := l.oGrads[b*l.featuresCount*l.iSize : (b+1)*l.featuresCount*l.iSize]

		for i, pos := range inputs {
			wGrads := l.wGrads[pos*l.featuresCount : (pos+1)*l.featuresCount]
			wGrads.Add(oGrads[i*l.featuresCount : (i+1)*l.featuresCount])
		}
	}
}

func (l *EmbedPos) ResetGrads() {
	l.oGrads.Fill(0)
	l.wGrads.Fill(0)
}

func (l *EmbedPos) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.wGrads}}
}
