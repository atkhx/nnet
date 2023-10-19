package metal

import (
	"context"
)

type Metal struct{}

func (d *Metal) GetSoftmaxTril(dst, src []float32, W, H, D int) Operation {
	return func(ctx context.Context) {
		d.SoftmaxTril(dst, src, W, H, D)
	}
}

func (d *Metal) SoftmaxTril(dst, src []float32, W, H, D int) {
	//cbuf := mps.CommandBufferFromContext(ctx)
	//for z := 0; z < output.Dims.D; z++ {
	//	cbuf.SoftmaxBufferTril(
	//		output.dataBuffer,
	//		aData.dataBuffer,
	//		//max.dataBuffer,
	//		//sum.dataBuffer,
	//		aData.Dims.W,
	//		aData.Dims.H,
	//		z*WH,
	//	)
	//}
}

func (d *Metal) GetDSoftmaxTril(aGrad, oGrad, softmax []float32, W, H, D int) Operation {
	return func(ctx context.Context) {
		d.DSoftmaxTril(aGrad, oGrad, softmax, W, H, D)
	}
}

func (d *Metal) DSoftmaxTril(aGrad, oGrad, softmax []float32, W, H, D int) {
	//cbuf := mps.CommandBufferFromContext(ctx)
	//for z := 0; z < output.Dims.D; z++ {
	//	cbuf.SoftmaxBufferTrilBwd(
	//		aData.gradBuffer,  // destination
	//		output.gradBuffer, // source
	//		output.dataBuffer, // softmax
	//		//softmaxGrad.dataBuffer, // softmaxGrad
	//		//sum.dataBuffer,         // sumOutBuffer
	//		aData.Dims.W, // colsCount
	//		aData.Dims.H, // rowsCount
	//		z*WH,         // offset
	//	)
	//}
}
