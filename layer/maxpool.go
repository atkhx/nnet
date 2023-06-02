package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewMaxPooling(iWidth, iHeight, filterSize, padding, stride int) *MaxPooling {
	if stride == 0 {
		stride = 1
	}

	return &MaxPooling{
		iWidth:     iWidth,
		iHeight:    iHeight,
		FilterSize: filterSize,
		Stride:     stride,
		Padding:    padding,
	}
}

type MaxPooling struct {
	iWidth, iHeight int

	FilterSize int
	Stride     int
	Padding    int

	outputObj *num.Data
}

func (l *MaxPooling) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.MaxPooling(l.iWidth, l.iHeight, l.FilterSize, l.Padding, l.Stride)
	return l.outputObj
}

func (l *MaxPooling) Forward() {
	l.outputObj.Forward()
}

func (l *MaxPooling) Backward() {
	l.outputObj.Backward()
}
