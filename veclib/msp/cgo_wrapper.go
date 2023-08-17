package msp

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "cgo_wrapper.h"
*/
import "C"
import "unsafe"

var DefaultDevice *MTLDevice

func InitDefaultDevice() {
	// todo Once
	if DefaultDevice == nil {
		DefaultDevice = NewMTLDevice()
	}
}

func ReleaseDefaultDevice() {
	// todo Once
	if DefaultDevice != nil {
		DefaultDevice.Release()
	}
}

type Releasable interface {
	Release()
}

func NewMTLDevice() *MTLDevice {
	return &MTLDevice{deviceID: unsafe.Pointer(C.createDevice())}
}

type MTLDevice struct {
	deviceID  unsafe.Pointer
	resources []Releasable
}

func (d *MTLDevice) regSource(source Releasable) {
	d.resources = append(d.resources, source)
}

func (d *MTLDevice) Release() {
	for i := len(d.resources); i > 0; i-- {
		d.resources[i].Release()
	}
	d.resources = nil
	C.releaseDevice(d.deviceID)
}

type MTLBuffer struct {
	id       unsafe.Pointer
	deviceID unsafe.Pointer
	contents []float32
	length   int
}

func (d *MTLDevice) CreateBufferWithBytes(data []float32) *MTLBuffer {
	bufferID := C.createNewBufferWithBytes(d.deviceID, (*C.float)(unsafe.Pointer(&data[0])), C.ulong(len(data)))
	contents := unsafe.Pointer(C.getBufferContents(bufferID))
	bfLength := len(data)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	buffer := &MTLBuffer{
		id:       bufferID,
		deviceID: d.deviceID,
		contents: float32Slice,
		length:   bfLength,
	}

	d.regSource(buffer)
	return buffer
}

func (b *MTLBuffer) GetData() []float32 {
	return b.contents
}

func (b *MTLBuffer) CopyFrom(source []float32) {
	copy(b.contents, source)
}

func (b *MTLBuffer) Release() {
	C.releaseBuffer(b.id)
}

func matrixMultiplyOnDevice(
	device *MTLDevice,
	a, b, c *MTLBuffer,
	aW int,
	alpha, beta float64,
) {
	aH := a.length / aW
	bH := aW

	bW := b.length / bH

	cW := bW
	cH := aH

	C.matrixMultiplyOnDevice(
		device.deviceID,
		a.id,
		b.id,
		c.id,
		C.int(aW), C.int(aH),
		C.int(bW), C.int(bH),
		C.int(cW), C.int(cH),
		C.int(aW),
		C.double(alpha),
		C.double(beta),
		C.bool(false),
		C.bool(false),
	)
}
