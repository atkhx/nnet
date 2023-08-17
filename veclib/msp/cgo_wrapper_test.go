package msp

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestNewDevice(t *testing.T) {
	device := NewMTLDevice()
	defer device.Release()

	aW, aH := 3, 3
	bW, bH := 2, aW

	bufferA := device.CreateBufferWithBytes(make([]float32, aW*aH))
	defer bufferA.Release()

	bufferB := device.CreateBufferWithBytes(make([]float32, bW*bH))
	defer bufferB.Release()

	bufferC := device.CreateBufferWithBytes(make([]float32, bW*aH))
	defer bufferC.Release()

	bufferAData := bufferA.GetData()
	for i := range bufferAData {
		bufferAData[i] = rand.Float32()
	}

	bufferBData := bufferB.GetData()
	for i := range bufferBData {
		bufferBData[i] = rand.Float32()
	}

	fmt.Println("bufferA.contents before MM", bufferA.contents)
	fmt.Println("bufferB.contents before MM", bufferB.contents)

	matrixMultiplyOnDevice(
		device,
		bufferA,
		bufferB,
		bufferC,
		3,
		1.0,
		0.0,
	)

	fmt.Println(bufferA.GetData())
	fmt.Println(bufferB.GetData())
	fmt.Println(bufferC.GetData())
}
