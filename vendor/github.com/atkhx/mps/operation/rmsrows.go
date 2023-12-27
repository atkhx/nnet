package operation

import "C"
import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/rmsrows"
)

func NewOpNormRMSByRows(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpNormRMSByRows {
	aggBuff := device.CreateBufferWithLength(inputData.Length / chunkSize)
	aggGrad := device.CreateBufferWithLength(inputData.Length / chunkSize)

	kernel := rmsrows.New(device.DeviceID)

	return &OpNormRMSByRows{
		device:      device,
		kernel:      kernel,
		aggTempData: aggBuff,
		aggTempGrad: aggGrad,
		inputData:   inputData,
		inputGrad:   inputGrad,
		outputData:  outputData,
		outputGrad:  outputGrad,
		chunkSize:   chunkSize,
	}
}

type OpNormRMSByRows struct {
	device *mps.MTLDevice
	kernel *rmsrows.Kernel

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	aggTempData *mps.MTLBuffer
	aggTempGrad *mps.MTLBuffer

	chunkSize int
}

func (op *OpNormRMSByRows) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.aggTempData.BufferID,
			op.chunkSize,
		)
	})
}

func (op *OpNormRMSByRows) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData.BufferID,
			op.inputGrad.BufferID,
			op.outputData.BufferID,
			op.outputGrad.BufferID,
			op.aggTempData.BufferID,
			op.aggTempGrad.BufferID,
			op.chunkSize,
		)
	})
}
