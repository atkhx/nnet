package lnorm

import (
	"github.com/atkhx/nnet/data"
)

func NewLayerNorm(H, D int) *LayerNorm {
	layer := &LayerNorm{}
	layer.gamma = data.NewData(1, H, D)
	layer.gamma.Data.Fill(1.0)

	layer.beta = data.NewData(1, H, D)
	layer.beta.Data.Fill(0.0)

	return layer
}

type LayerNorm struct {
	gamma *data.Data
	beta  *data.Data
}

func (l *LayerNorm) Forward(inputs *data.Data) (output *data.Data) {
	eps := 1e-5

	meanByRows := inputs.RowMean()
	varsByRows := inputs.RowVariance()

	varsByRows.Data.AddScalar(eps)
	xsubMean := inputs.SubColVector(meanByRows)

	xhat := xsubMean.DivColVector(varsByRows.Sqrt())

	return xhat.MulColVector(l.gamma).AddColVector(l.beta)
}
