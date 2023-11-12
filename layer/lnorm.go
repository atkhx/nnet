package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewLNorm() *LNorm {
	return &LNorm{}
}

type LNorm struct {
	Gamma *num.Data
	Beta  *num.Data

	forUpdate []*num.Data
}

func (l *LNorm) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	panic("not implemented")
	//rowWidth := device.GetDataDims(inputs).W
	//
	//l.Beta = device.NewData(num.NewDims(rowWidth))
	//l.Gamma = device.NewData(num.NewDims(rowWidth))
	//
	//device.FillDataWithOnes(l.Gamma)
	//
	//l.forUpdate = []*num.Data{l.Gamma, l.Beta}
	//
	//return device.LNorm(inputs, l.Gamma, l.Beta)
}

func (l *LNorm) ForUpdate() []*num.Data {
	return l.forUpdate
}
