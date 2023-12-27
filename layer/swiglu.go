package layer

import (
	"encoding/json"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSwiGLU(featuresCount, hiddenSize int, initWeights initializer.Initializer, provideWeights func(w1, w2, w3 *num.Data)) *SwiGLU {
	return &SwiGLU{featuresCount: featuresCount, hiddenSize: hiddenSize, initWeights: initWeights, provideWeights: provideWeights}
}

type SwiGLU struct {
	initWeights    initializer.Initializer
	provideWeights func(w1, w2, w3 *num.Data)

	hiddenSize    int
	featuresCount int

	weights1 *num.Data
	weights2 *num.Data
	weights3 *num.Data

	forUpdate []*num.Data
}

func (l *SwiGLU) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))
	inputWidth := device.GetDataDims(inputs).W

	l.weights1 = device.NewDataRandNormWeighted(num.NewDims(l.hiddenSize, inputWidth), weightK)
	l.weights2 = device.NewDataRandNormWeighted(num.NewDims(l.hiddenSize, inputWidth), weightK)
	l.weights3 = device.NewDataRandNormWeighted(num.NewDims(l.featuresCount, l.hiddenSize), weightK)

	w1Projection := device.MatrixMultiply(inputs, l.weights1, 1)
	w2Projection := device.MatrixMultiply(inputs, l.weights2, 1)

	w1SiLU := device.SiLu(w1Projection)

	l.forUpdate = []*num.Data{l.weights1, l.weights2, l.weights3}

	return device.MatrixMultiply(device.MulEqual(w1SiLU, w2Projection), l.weights3, 1)
}

func (l *SwiGLU) ForUpdate() []*num.Data {
	return l.forUpdate
}

type SwiGLUConfig struct {
	Weights1 []float32
	Weights2 []float32
	Weights3 []float32
}

func (l *SwiGLU) MarshalJSON() ([]byte, error) {
	return json.Marshal(SwiGLUConfig{
		Weights1: l.weights1.Data.GetData(),
		Weights2: l.weights2.Data.GetData(),
		Weights3: l.weights3.Data.GetData(),
	})
}

func (l *SwiGLU) UnmarshalJSON(bytes []byte) error {
	config := SwiGLUConfig{
		Weights1: l.weights1.Data.GetData(),
		Weights2: l.weights2.Data.GetData(),
		Weights3: l.weights3.Data.GetData(),
	}
	return json.Unmarshal(bytes, &config)
}

func (l *SwiGLU) LoadFromProvider() {
	l.provideWeights(l.weights1, l.weights2, l.weights3)
}
