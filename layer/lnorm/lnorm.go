package lnorm

import (
	"github.com/atkhx/nnet/data"
)

func NewLayerNorm(batchSize int) *LayerNorm {
	layer := &LayerNorm{}
	layer.gamma = data.NewData(1, batchSize, 1)
	layer.gamma.Data.Fill(1.0)

	layer.beta = data.NewData(1, batchSize, 1)
	layer.beta.Data.Fill(0.0)

	layer.batchSize = batchSize

	return layer
}

type LayerNorm struct {
	gamma     *data.Data
	beta      *data.Data
	batchSize int
}

func (l *LayerNorm) Forward(inputs *data.Data) *data.Data {
	//eps := 1e-5
	W, H, D := inputs.Data.W, inputs.Data.H, inputs.Data.D

	inputs = inputs.Reshape(inputs.Data.GetLen()/l.batchSize, l.batchSize, 1)

	meanByRows := inputs.MeanByRows()
	varsByRows := inputs.RowVariance()

	//varsByRows.Data.AddScalar(eps)
	xsubMean := inputs.SubColVector(meanByRows)

	xhat := xsubMean.DivColVector(varsByRows.Sqrt())

	return xhat.MulColVector(l.gamma).AddColVector(l.beta).Reshape(W, H, D)
}