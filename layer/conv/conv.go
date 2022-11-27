package conv

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/executor"
)

func New(options ...Option) *Conv {
	layer := &Conv{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)
	return layer
}

type Conv struct {
	FCount   int
	FPadding int
	FStride  int

	Weights *data.Data
	Biases  *data.Data

	Trainable bool

	iWidth, iHeight, iDepth int
	FWidth, FHeight, FDepth int
	oWidth, oHeight, oDepth int

	InitWeightsParams InitWeightsParams

	inputs *data.Data
	output *data.Data

	iGrads *data.Data
	wGrads *data.Data
	bGrads *data.Data

	iSquare int
	oSquare int
	fSquare int

	iCube int
	fCube int

	oHiW int
	fHiW int

	iGradsSeparated [][]float64
}

func (l *Conv) InitDataSizes(iw, ih, id int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = iw+2*l.FPadding, ih+2*l.FPadding, id

	l.oWidth = (iw-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (ih-l.FHeight+2*l.FPadding)/l.FStride + 1

	l.oDepth = l.FCount
	l.FDepth = id

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.fSquare = l.FWidth * l.FHeight

	l.fCube = l.FDepth * l.fSquare
	l.iCube = l.iDepth * l.iSquare

	l.oHiW = l.oHeight * l.iWidth
	l.fHiW = l.FHeight * l.iWidth

	if l.Weights == nil {
		l.Weights = &data.Data{}
		l.Biases = &data.Data{}
	}

	if len(l.Weights.Data) == 0 {
		l.Weights.Init3DRandom(
			l.FWidth,
			l.FHeight,
			l.FCount*l.FDepth,
			l.InitWeightsParams.WeightMinThreshold,
			l.InitWeightsParams.WeightMaxThreshold,
		)
		l.Biases.InitVector(l.FCount)
		l.Biases.Fill(l.InitWeightsParams.BiasInitialValue)
	}

	l.output = &data.Data{}
	l.output.Init3D(l.oWidth, l.oHeight, l.oDepth)

	l.bGrads = &data.Data{}
	l.bGrads.InitVector(l.FCount)

	l.wGrads = &data.Data{}
	l.wGrads.Init3D(l.FWidth, l.FHeight, l.FCount*l.FDepth)

	l.iGrads = &data.Data{}
	l.iGrads.Init3D(l.iWidth, l.iHeight, l.iDepth)

	l.iGradsSeparated = make([][]float64, l.FCount)
	for i := 0; i < len(l.iGradsSeparated); i++ {
		l.iGradsSeparated[i] = make([]float64, l.iCube)
	}

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Conv) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs.AddPadding(l.FPadding)

	executor.RunParallel(l.FCount, func(filterIndex int) {
		outputOffset := filterIndex * l.oSquare
		filterOffset := filterIndex * l.fCube

		inputs := l.inputs.Data
		output := l.output.Data[outputOffset : outputOffset+l.oSquare]
		filter := l.Weights.Data[filterOffset : filterOffset+l.fCube]

		data.Fill(output, l.Biases.Data[filterIndex])

		wCoord := 0
		for izo := 0; izo < l.iCube; izo += l.iSquare {
			for iyo := izo; iyo < izo+l.fHiW; iyo += l.iWidth {
				for ixo := iyo; ixo < iyo+l.FWidth; ixo++ {
					weight := filter[wCoord]
					wCoord++

					oCoord := 0
					for iCoord := ixo; iCoord < ixo+l.oHiW; iCoord += l.iWidth {
						output := output[oCoord : oCoord+l.oWidth]
						inputs := inputs[iCoord : iCoord+l.oWidth]
						for ic, iv := range inputs {
							output[ic] += iv * weight
						}
						oCoord += l.oWidth
					}
				}
			}
		}
	})

	return l.output
}

func (l *Conv) Backward(deltas *data.Data) *data.Data {
	l.iGrads.FillZero()

	executor.RunParallel(l.FCount, func(filterIndex int) {
		outputOffset := filterIndex * l.oSquare
		filterOffset := filterIndex * l.fCube

		inputs := l.inputs.Data
		filter := l.Weights.Data[filterOffset : filterOffset+l.fCube]
		deltas := deltas.Data[outputOffset : outputOffset+l.oSquare]
		wGrads := l.wGrads.Data[filterOffset : filterOffset+l.fCube]

		l.bGrads.Data[filterIndex] += data.SumElements(deltas)

		iGrads := l.iGradsSeparated[filterIndex]
		copy(iGrads, l.iGrads.Data)

		wCoord := 0
		for izo := 0; izo < l.iCube; izo += l.iSquare {
			for iyo := izo; iyo < izo+l.fHiW; iyo += l.iWidth {
				for ixo := iyo; ixo < iyo+l.FWidth; ixo++ {
					wgradv := wGrads[wCoord]
					weight := filter[wCoord]
					oCoord := 0

					for iCoord := ixo; iCoord < ixo+l.oHiW; iCoord += l.iWidth {
						inputs := inputs[iCoord : iCoord+l.oWidth]
						iGrads := iGrads[iCoord : iCoord+l.oWidth]
						deltas := deltas[oCoord : oCoord+l.oWidth]

						for dc, delta := range deltas {
							iGrads[dc] += delta * weight
							wgradv += inputs[dc] * delta
						}

						oCoord += l.oWidth
					}

					wGrads[wCoord] = wgradv
					wCoord++
				}
			}
		}
	})

	l.iGrads.Add(l.iGradsSeparated...)
	return l.iGrads.RemovePadding(l.FPadding)
}

func (l *Conv) ResetGradients() {
	l.wGrads.FillZero()
	l.bGrads.FillZero()
}

func (l *Conv) GetWeights() *data.Data {
	return l.Weights
}

func (l *Conv) GetOutput() *data.Data {
	return l.output
}

func (l *Conv) GetInputs() *data.Data {
	return l.inputs
}

func (l *Conv) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.Weights, l.wGrads
}

func (l *Conv) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.bGrads
}

func (l *Conv) GetInputGradients() *data.Data {
	return l.iGrads
}

func (l *Conv) GetWeightGradients() *data.Data {
	return l.wGrads
}

func (l *Conv) IsTrainable() bool {
	return l.Trainable
}
