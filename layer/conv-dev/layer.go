package conv_dev

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/executor"
)

func New(options ...Option) *Layer {
	layer := &Layer{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	return layer
}

type Layer struct {
	// begin storable Layer config
	FCount   int
	FPadding int
	FStride  int

	Weights *data.Data
	Biases  *data.Data

	Trainable bool
	// end storable Layer config

	iWidth, iHeight, iDepth int
	FWidth, FHeight, FDepth int
	oWidth, oHeight, oDepth int

	initWeights InitWeightsParams

	inputs *data.Data
	output *data.Data

	gradInputs  *data.Data
	gradWeights *data.Data
	gradBiases  *data.Data

	iSquare int
	oSquare int
	fSquare int

	iCube int
	fCube int

	oHiW int
	fHiW int

	gradInputsSeparated [][]float64
}

func (l *Layer) InitDataSizes(iw, ih, id int) (int, int, int) {
	//l.iWidth, l.iHeight, l.iDepth = iw, ih, id
	l.iWidth, l.iHeight, l.iDepth = iw+2*l.FPadding, ih+2*l.FPadding, id

	//fmt.Println("iw, ih, id", iw, ih, id)
	//fmt.Println("iw, ih, id", l.iWidth, l.iHeight, l.iDepth, "padded")

	l.oWidth = (iw-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (ih-l.FHeight+2*l.FPadding)/l.FStride + 1

	l.oDepth = l.FCount
	l.FDepth = id

	//fmt.Println("fw, fh, fd", l.FWidth, l.FHeight, l.FDepth)

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
		l.Weights.InitCubeRandom(
			l.FWidth,
			l.FHeight,
			l.FCount*l.FDepth,
			l.initWeights.WeightMinThreshold,
			l.initWeights.WeightMaxThreshold,
		)
		l.Biases.InitVector(l.FCount)
		l.Biases.Fill(l.initWeights.BiasInitialValue)
	}

	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradBiases = &data.Data{}
	l.gradBiases.InitVector(l.FCount)

	l.gradWeights = &data.Data{}
	l.gradWeights.InitCube(l.FWidth, l.FHeight, l.FCount*l.FDepth)

	l.gradInputs = &data.Data{}
	//l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)
	l.gradInputs.InitCube(iw, ih, id)

	l.gradInputsSeparated = make([][]float64, l.FCount)
	for i := 0; i < len(l.gradInputsSeparated); i++ {
		//l.gradInputsSeparated[i] = make([]float64, l.iCube)
		l.gradInputsSeparated[i] = make([]float64, iw*ih*id)
	}

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Layer) Activate(inputs *data.Data) *data.Data {
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
					for iOffset := ixo; iOffset < ixo+l.oHiW; iOffset += l.iWidth {
						output := output[oCoord : oCoord+l.oWidth]
						inputs := inputs[iOffset : iOffset+l.oWidth]
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

func (l *Layer) Backprop(del *data.Data) *data.Data {
	l.gradInputs.FillZero()

	// calc weight gradients
	executor.RunParallel(l.FCount, func(filterIndex int) {
		outputOffset := filterIndex * l.oSquare
		filterOffset := filterIndex * l.fCube

		inputs := l.inputs.Data
		deltas := del.Data[outputOffset : outputOffset+l.oSquare]
		wgrads := l.gradWeights.Data[filterOffset : filterOffset+l.fCube]

		l.gradBiases.Data[filterIndex] += data.SumElements(deltas)

		deltaCoord := 0
		for iyo := 0; iyo < l.oHiW; iyo += l.iWidth { // deltas height (l.oHeight times)
			for ixo := iyo; ixo < iyo+l.oWidth; ixo++ { // deltas width (l.oWidth times)

				delta := deltas[deltaCoord]
				deltaCoord++

				wgCoord := 0

				for izo := ixo; izo < ixo+l.iCube; izo += l.iSquare { // inputs depth = weights depth (l.iDepth times)
					for iOffset := izo; iOffset < izo+(l.FHeight*l.iWidth); iOffset += l.iWidth {
						inputs := inputs[iOffset : iOffset+l.FWidth]
						wgrads := wgrads[wgCoord : wgCoord+l.FWidth]

						for ic, iv := range inputs {
							wgrads[ic] += iv * delta
						}

						wgCoord += l.FWidth
					}
				}
			}
		}
	})

	var dWidth, dHeight, dDepth int
	del.ExtractDimensions(&dWidth, &dHeight, &dDepth)

	dPadding := ((l.iHeight-2*l.FPadding-1)*l.FStride + l.FHeight - dHeight) / 2
	deltas := del.AddPadding(dPadding)
	deltas.ExtractDimensions(&dWidth, &dHeight, &dDepth)

	dSquare := dWidth * dHeight
	weightsRotated := l.Weights.Rotate180()

	originalIWidth := l.iWidth - 2*l.FPadding
	originalISquare := originalIWidth * (l.iHeight - 2*l.FPadding)

	// calc input gradients
	executor.RunParallel(l.FCount, func(filterIndex int) {
		deltasOffset := filterIndex * dSquare
		filterOffset := filterIndex * l.fCube

		deltas := deltas.Data[deltasOffset : deltasOffset+dSquare]
		filter := weightsRotated.Data[filterOffset : filterOffset+l.fCube]

		igrads := l.gradInputsSeparated[filterIndex]
		copy(igrads, l.gradInputs.Data)

		wCoord := 0
		for fz := 0; fz < l.FDepth; fz++ {
			for fy := 0; fy < l.FHeight; fy++ {
				for fx := 0; fx < l.FWidth; fx++ {
					//weight := filter[fz*l.fSquare+fy*l.FWidth+fx]
					weight := filter[wCoord]
					wCoord++

					for dy := 0; dy < dHeight-l.FHeight+1; dy++ {
						dOffset := (dy+fy)*dWidth + fx
						iOffset := fz*originalISquare + dy*originalIWidth

						deltas := deltas[dOffset : dOffset+dWidth-l.FWidth+1]
						igrads := igrads[iOffset : iOffset+dWidth-l.FWidth+1]

						for ic, delta := range deltas {
							igrads[ic] += delta * weight
						}
					}
				}
			}
		}
	})

	l.gradInputs.Add(l.gradInputsSeparated...)
	return l.gradInputs
}

func (l *Layer) ResetGradients() {
	l.gradWeights.FillZero()
	l.gradBiases.FillZero()
}

func (l *Layer) GetWeights() *data.Data {
	return l.Weights
}

func (l *Layer) GetOutput() *data.Data {
	return l.output
}

func (l *Layer) GetInputs() *data.Data {
	return l.inputs
}

func (l *Layer) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.Weights, l.gradWeights
}

func (l *Layer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.gradBiases
}

func (l *Layer) GetInputGradients() *data.Data {
	return l.gradInputs
}

func (l *Layer) GetWeightGradients() *data.Data {
	return l.gradWeights
}

func (l *Layer) IsTrainable() bool {
	return l.Trainable
}
