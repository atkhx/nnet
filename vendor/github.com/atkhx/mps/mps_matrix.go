package mps

import (
	"unsafe"

	"github.com/atkhx/mps/framework"
)

func NewMPSMatrix(bufferID unsafe.Pointer, cols, rows, batchSize, batchStride, offset int) *MPSMatrix {
	descriptorID := framework.MPSMatrixDescriptorCreate(cols, rows, batchSize, batchStride)
	matrixID := framework.MPSMatrixCreate(bufferID, descriptorID, offset)

	return &MPSMatrix{
		MatrixID:     matrixID,
		DescriptorID: descriptorID,
	}
}

type MPSMatrix struct {
	MatrixID     unsafe.Pointer
	DescriptorID unsafe.Pointer
}

func (m *MPSMatrix) Release() {
	framework.MPSMatrixDescriptorRelease(m.DescriptorID)
	framework.MPSMatrixRelease(m.MatrixID)
}
