package operation

import (
	"time"
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/framework"
	"github.com/atkhx/mps/operation/dropout"
)

func NewOpDropout(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	probability float32,
) *OpDropout {
	distribution := device.CreateMatrixRandomDistribution(0, 1)
	randomizer := device.CreateMatrixRandomMTGP32(distribution, uint64(time.Now().UnixNano()))

	maskBuffer := device.CreateBufferWithLength(inputData.Length)
	maskMatrix := maskBuffer.CreateMatrix(inputData.Length, 1, 0)

	return &OpDropout{
		kernel: dropout.New(device.DeviceID),

		randomizer: randomizer,

		maskBuffer: maskBuffer,
		maskMatrix: maskMatrix,

		inputData: inputData.BufferID,
		inputGrad: inputGrad.BufferID,

		outputData: outputData.BufferID,
		outputGrad: outputGrad.BufferID,

		probability: probability,
	}
}

type OpDropout struct {
	kernel     *dropout.Kernel
	randomizer *mps.MatrixRandomMTGP32

	maskBuffer *mps.MTLBuffer
	maskMatrix *mps.MPSMatrix

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	probability float32
}

func (op *OpDropout) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		framework.MPSMatrixRandomMTGP32Encode(op.randomizer.ID, b.ID, op.maskMatrix.MatrixID)

		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.maskBuffer.BufferID,
			op.probability,
		)
	})
}

func (op *OpDropout) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.outputGrad,
			op.maskBuffer.BufferID,
			op.probability,
		)
	})
}
