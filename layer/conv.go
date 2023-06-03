package layer

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewConv(
	filterSize int,
	filtersCount int,
	padding int,
	stride int,
	imageWidth int,
	imageHeight int,
	initWeights initializer.Initializer,
) *Conv {
	if stride == 0 {
		stride = 1
	}

	return &Conv{
		filterSize:   filterSize,
		filtersCount: filtersCount,
		padding:      padding,
		stride:       stride,
		imageWidth:   imageWidth,
		imageHeight:  imageHeight,
		initWeights:  initWeights,
	}
}

type Conv struct {
	filterSize   int
	filtersCount int

	padding int
	stride  int

	imageWidth  int
	imageHeight int

	initWeights initializer.Initializer

	WeightObj *num.Data
	BiasesObj *num.Data

	inputsObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Conv) Compile(inputs *num.Data) *num.Data {
	channels := inputs.Dims.H

	weightK := l.initWeights.GetNormK(l.filterSize * l.filterSize * channels)

	l.WeightObj = num.NewRandNormWeighted(num.NewDims(l.filterSize*l.filterSize, channels, l.filtersCount), weightK)
	l.BiasesObj = num.New(num.NewDims(1, 1, l.filtersCount))

	l.outputObj = inputs.Conv(l.imageWidth, l.imageHeight, l.filterSize, l.padding, l.stride, l.WeightObj, l.BiasesObj)
	l.forUpdate = num.Nodes{l.WeightObj, l.BiasesObj}

	l.inputsObj = inputs
	return l.outputObj
}

func (l *Conv) Forward() {
	l.outputObj.Forward()
}

func (l *Conv) Backward() {
	l.outputObj.Backward()
}

func (l *Conv) ForUpdate() num.Nodes {
	return l.forUpdate
}

func (l *Conv) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Conv) GetOutput() *num.Data {
	return l.outputObj
}
