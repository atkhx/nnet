package mps

import (
	"unsafe"
)

var DefaultDevice *MTLDevice

func InitDefaultDevice() {
	if DefaultDevice == nil {
		DefaultDevice = NewMTLDevice()
	}
}

func ReleaseDefaultDevice() {
	if DefaultDevice != nil {
		DefaultDevice.Release()
	}
}

type Releasable interface {
	Release()
}

func NewMTLDevice() *MTLDevice {
	deviceID := mtlDeviceCreate()
	device := &MTLDevice{
		deviceID: deviceID,

		krnFill:           customKernelFillCreate(deviceID),
		krnReLUFwd:        customKernelReLUForwardCreate(deviceID),
		krnReLUBwd:        customKernelReLUBackwardCreate(deviceID),
		krnMul:            customKernelMulCreate(deviceID),
		krnDropout:        customKernelDropoutCreate(deviceID),
		krnSoftmax:        customKernelSoftmaxForwardCreate(deviceID),
		krnSoftmaxTrilFwd: customKernelSoftmaxTrilForwardCreate(deviceID),
		krnSoftmaxTrilBwd: customKernelSoftmaxTrilBackwardCreate(deviceID),
		krnUpdateWithAdam: customKernelUpdateWithAdamCreate(deviceID),
	}

	return device
}

type MTLDevice struct {
	deviceID  unsafe.Pointer
	resources []Releasable

	krnFill           unsafe.Pointer
	krnReLUFwd        unsafe.Pointer
	krnReLUBwd        unsafe.Pointer
	krnMul            unsafe.Pointer
	krnDropout        unsafe.Pointer
	krnSoftmax        unsafe.Pointer
	krnSoftmaxTrilFwd unsafe.Pointer
	krnSoftmaxTrilBwd unsafe.Pointer
	krnUpdateWithAdam unsafe.Pointer
}

func (device *MTLDevice) CreateCommandQueue() *MTLCommandQueue {
	queue := NewMTLCommandQueue(device)
	device.regSource(queue)
	return queue
}

func (device *MTLDevice) CreateBufferWithBytes(data []float32) *MTLBuffer {
	buffer := NewMTLBufferWithBytes(device, data)
	device.regSource(buffer)
	return buffer
}

func (device *MTLDevice) CreateBufferWithLength(bfLength int) *MTLBuffer {
	buffer := NewMTLBufferWithLength(device, bfLength)
	device.regSource(buffer)
	return buffer
}

func (device *MTLDevice) regSource(source Releasable) {
	device.resources = append(device.resources, source)
}

func (device *MTLDevice) Release() {
	for i := len(device.resources); i > 0; i-- {
		device.resources[i-1].Release()
	}
	device.resources = nil
	mtlDeviceRelease(device.deviceID)
}
