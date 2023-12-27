package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/mulcols"
)

func NewOpMulCols(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	weightsData *mps.MTLBuffer,
	weightsGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	rowWidth int,
	colHeight int,
) *OpMulCols {
	return &OpMulCols{
		kernel: mulcols.New(device.DeviceID),

		inputData:   inputData.BufferID,
		inputGrad:   inputGrad.BufferID,
		weightsData: weightsData.BufferID,
		weightsGrad: weightsGrad.BufferID,
		outputData:  outputData.BufferID,
		outputGrad:  outputGrad.BufferID,

		rowWidth:  rowWidth,
		colHeight: colHeight,
	}
}

type OpMulCols struct {
	kernel *mulcols.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	weightsData unsafe.Pointer
	weightsGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	rowWidth  int
	colHeight int
}

func (op *OpMulCols) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.weightsData,
			op.outputData,
			op.rowWidth,
			op.colHeight,
		)
	})
}

func (op *OpMulCols) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData,
			op.inputGrad,
			op.weightsData,
			op.weightsGrad,
			op.outputData,
			op.outputGrad,
			op.rowWidth,
			op.colHeight,
		)
	})
}
