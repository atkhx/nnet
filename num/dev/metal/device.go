package metal

import (
	"context"
	"math"
	"math/rand"

	"github.com/atkhx/mps"
	ops "github.com/atkhx/mps/operation"
	"github.com/atkhx/nnet/num"
)

type Device struct {
	mtlDevice *mps.MTLDevice
}

func NewDevice() *Device {
	return &Device{mtlDevice: mps.NewMTLDevice()}
}

func (d *Device) Release() {
	d.mtlDevice.Release()
}

func (d *Device) CreateCommandQueue() *mps.MTLCommandQueue {
	return d.mtlDevice.CreateCommandQueue()
}

func (d *Device) NewData(dims num.Dims, srcNodes ...*num.Data) *num.Data {
	return num.NewData(
		d.mtlDevice.CreateBufferWithLength(dims.Size()),
		d.mtlDevice.CreateBufferWithLength(dims.Size()),
		dims,
		srcNodes...,
	)
}

func (d *Device) NewDataWithValues(dims num.Dims, values []float32) *num.Data {
	if dims.Size() != len(values) {
		panic("invalid values length")
	}
	return num.NewData(
		d.mtlDevice.CreateBufferWithBytes(values),
		d.mtlDevice.CreateBufferWithLength(dims.Size()),
		dims,
	)
}

func (d *Device) newLinkedCopy(data *num.Data, links ...*num.Data) *num.Data {
	return d.NewData(data.Dims, append([]*num.Data{data}, links...)...)
}

func (d *Device) NewDataRandNormWeighted(dims num.Dims, w float32) *num.Data {
	data := make([]float32, dims.Size())
	for i := range data {
		data[i] = float32(rand.NormFloat64()) * w
	}

	return d.NewDataWithValues(dims, data)
}

func (d *Device) NewTokenEmbeddingTable(featuresCount, alphabetSize int) *num.Data {
	dims := num.NewDims(featuresCount, alphabetSize)
	data := make([]float32, dims.Size())
	for i := range data {
		data[i] = float32(rand.NormFloat64())
	}

	return d.NewDataWithValues(dims, data)
}

func (d *Device) NewPositionEmbeddingTable(featuresCount, contextSize int) *num.Data {
	values := make([]float32, featuresCount*contextSize)

	k := 0
	for j := 0; j < contextSize; j++ {
		for i := 0; i < featuresCount; i++ {
			if i%2 == 0 {
				values[k] = float32(math.Sin(float64(k) / math.Pow(10_000, float64(i+1)/float64(featuresCount))))
			} else {
				values[k] = float32(math.Cos(float64(k) / math.Pow(10_000, float64(i+1)/float64(featuresCount))))
			}
			k++
		}
	}
	return d.NewDataWithValues(num.NewDims(featuresCount, contextSize), values)
}

func (d *Device) GetDataDims(data *num.Data) num.Dims {
	return data.Dims
}

func (d *Device) GetDataLength(data *num.Data) int {
	return data.Data.Length
}

func (d *Device) GetData(node *num.Data) []float32 {
	return node.Data.GetData()
}

func (d *Device) Mean(input *num.Data) *num.Data {
	output := d.NewData(num.NewDims(), input)
	operation := ops.NewOpMean(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		input.Dims.Size(),
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) Add(aData, bData *num.Data) *num.Data {
	if aData.Dims == bData.Dims {
		return d.AddEqual(aData, bData)
	}
	if aData.Dims.W == bData.Dims.W && bData.Dims.H == 1 && bData.Dims.D == 1 {
		return d.AddRow(aData, bData, aData.Dims.W)
	}
	panic("not implemented")
}

func (d *Device) AddRow(input, weights *num.Data, width int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	operation := ops.NewOpAddRows(
		d.mtlDevice,
		input.Data,
		input.Grad,
		weights.Data,
		weights.Grad,
		output.Data,
		output.Grad,
		width,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) AddEqual(input, weights *num.Data) *num.Data {
	if input.Dims != weights.Dims {
		panic("dimensions are not equal")
	}
	output := d.newLinkedCopy(input, weights)
	operation := ops.NewOpAddEqual(
		d.mtlDevice,
		input.Data,
		input.Grad,
		weights.Data,
		weights.Grad,
		output.Data,
		output.Grad,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) MulCol(input, weights *num.Data, width, height int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	operation := ops.NewOpMulCols(
		d.mtlDevice,
		input.Data,
		input.Grad,
		weights.Data,
		weights.Grad,
		output.Data,
		output.Grad,
		width,
		height,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) MulRow(input, weights *num.Data, width int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	operation := ops.NewOpMulRows(
		d.mtlDevice,
		input.Data,
		input.Grad,
		weights.Data,
		weights.Grad,
		output.Data,
		output.Grad,
		width,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) MulEqual(input, weights *num.Data) *num.Data {
	if input.Dims.Size() != weights.Dims.Size() {
		panic("input size != weights size")
	}
	output := d.newLinkedCopy(input, weights)
	operation := ops.NewOpMulEqual(
		d.mtlDevice,
		input.Data,
		input.Grad,
		weights.Data,
		weights.Grad,
		output.Data,
		output.Grad,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) RMSNorm(input *num.Data, width int) *num.Data {
	output := d.newLinkedCopy(input)
	operation := ops.NewOpRMSNormByRows(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		width,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) Rope(input *num.Data, headIndex, headSize, contextLength int) *num.Data {
	output := d.newLinkedCopy(input)
	operation := ops.NewOpRope(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		headIndex,
		headSize,
		contextLength,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) RopeCols(input *num.Data, featuresCount, headSize, contextLength int) *num.Data {
	output := d.newLinkedCopy(input)
	operation := ops.NewOpRopeCols(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		featuresCount,
		headSize,
		contextLength,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) RopeRows(input *num.Data, featuresCount, headSize, contextLength int) *num.Data {
	output := d.newLinkedCopy(input)
	operation := ops.NewOpRopeRows(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		featuresCount,
		headSize,
		contextLength,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) Relu(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	operation := ops.NewOpReLu(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) SiLu(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	operation := ops.NewOpSiLu(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) ConcatByRows(bData ...*num.Data) *num.Data {
	dims := bData[0].Dims

	output := d.NewData(num.NewDims(dims.W*len(bData), dims.H, dims.D), bData...)

	inputDataBuffers := []*mps.MTLBuffer{}
	inputGradBuffers := []*mps.MTLBuffer{}

	for _, node := range bData {
		inputDataBuffers = append(inputDataBuffers, node.Data)
		inputGradBuffers = append(inputGradBuffers, node.Grad)
	}

	operation := ops.NewOpConcatByRows(
		d.mtlDevice,
		inputDataBuffers,
		inputGradBuffers,
		output.Data,
		output.Grad,
		dims.W,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) Dropout(input *num.Data, prob float32) *num.Data {
	if prob == 0 {
		return input
	}

	output := d.newLinkedCopy(input)
	operation := ops.NewOpDropout(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		prob,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) Reshape(input *num.Data, dims num.Dims) *num.Data {
	if input.Dims.Size() != dims.Size() {
		panic("total dimension size must be equal with original")
	}

	output := *input
	output.Dims = dims
	output.SrcNodes = num.Nodes{input}
	output.SkipResetGrad = true
	output.CalcData = func(_ context.Context) {}
	output.CalcGrad = func(_ context.Context) {}
	return &output
}

func (d *Device) CrossEntropyPos(input, targets *num.Data) *num.Data {
	if targets.Dims.W != 1 {
		panic("target width must be equal 1")
	}

	if targets.Dims.H != input.Dims.H {
		panic("target height must be equal input height")
	}

	if targets.Dims.D != input.Dims.D {
		panic("target depth must be equal input depth")
	}

	output := d.NewData(targets.Dims, input)

	operation := ops.NewOpCrossEntropy(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		targets.Data,
		input.Dims.W,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}

	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}

	return output
}

func (d *Device) Embeddings(input *num.Data, tEmbeddings *num.Data) *num.Data {
	featuresCount := tEmbeddings.Dims.W

	contextSize := input.Dims.W
	tokensCount := input.Dims.H

	output := d.NewData(num.NewDims(featuresCount, contextSize, tokensCount), tEmbeddings)

	operation := ops.NewOpEmbeddings(
		d.mtlDevice,
		tEmbeddings.Data,
		tEmbeddings.Grad,
		input.Data,
		output.Data,
		output.Grad,
		featuresCount,
		contextSize,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) TokenPosEmbeddings(input *num.Data, tEmbeddings, pEmbeddings *num.Data) *num.Data {
	// pos + token embeddings in one op
	if tEmbeddings.Dims.W != pEmbeddings.Dims.W {
		panic("features count must be equal")
	}

	featuresCount := tEmbeddings.Dims.W

	contextSize := input.Dims.W
	tokensCount := input.Dims.H

	output := d.NewData(num.NewDims(featuresCount, contextSize, tokensCount), tEmbeddings)

	operation := ops.NewOpPosEmbeddings(
		d.mtlDevice,
		tEmbeddings.Data,
		tEmbeddings.Grad,
		pEmbeddings.Data,
		input.Data,
		output.Data,
		output.Grad,
		featuresCount,
		contextSize,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) Transpose(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	output.Dims.W = input.Dims.H
	output.Dims.H = input.Dims.W

	operation := ops.NewOpTranspose(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		input.Dims.W,
		input.Dims.H,
	)
	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) TriangleLowerSoftmax(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)

	operation := ops.NewOpTriangularLowedSoftmax3(
		d.mtlDevice,
		input.Data,
		input.Grad,
		output.Data,
		output.Grad,
		input.Dims.W,
		input.Dims.H,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}
	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}
	return output
}

func (d *Device) MatrixMultiply(aData, bData *num.Data, alpha float32) *num.Data {
	if aData.Dims.W != bData.Dims.H {
		panic("aData width must be equal bData height")
	}

	if aData.Dims.D != bData.Dims.D && !(aData.Dims.D == 1 || bData.Dims.D == 1) {
		panic("aData's and bData's dept must be equal or one of them must be 1")
	}

	oD := aData.Dims.D
	if bData.Dims.D > oD {
		oD = bData.Dims.D
	}

	oW := bData.Dims.W
	oH := aData.Dims.H

	output := d.NewData(num.Dims{W: oW, H: oH, D: oD}, aData, bData)

	operation := ops.NewOpMatrixMultiply(
		d.mtlDevice,
		aData.Data,
		aData.Grad,

		bData.Data,
		bData.Grad,

		output.Data,
		output.Grad,

		aData.Dims.W, aData.Dims.H, aData.Dims.D,
		bData.Dims.W, bData.Dims.H, bData.Dims.D,

		output.Dims.W, output.Dims.H, output.Dims.D,

		alpha,
	)

	output.CalcData = func(ctx context.Context) {
		operation.Forward(mps.CommandBufferFromContext(ctx))
	}

	output.CalcGrad = func(ctx context.Context) {
		operation.Backward(mps.CommandBufferFromContext(ctx))
	}

	return output
}

func (d *Device) GetOptimizerAdam(iterations int, beta1, beta2, learningRate, eps float32) func(nodes []*num.Data) func(ctx context.Context, iteration int) {
	iterations++
	return func(nodes []*num.Data) func(ctx context.Context, iteration int) {

		mm := make([]*mps.MTLBuffer, len(nodes))
		vv := make([]*mps.MTLBuffer, len(nodes))

		for i, node := range nodes {
			mm[i] = d.mtlDevice.CreateBufferWithLength(d.GetDataLength(node))
			vv[i] = d.mtlDevice.CreateBufferWithLength(d.GetDataLength(node))
		}

		beta1pow := make([]float32, iterations)
		beta2pow := make([]float32, iterations)

		for i := 0; i < iterations; i++ {
			if i == 0 {
				beta1pow[i] = beta1
				beta2pow[i] = beta2
			} else {
				beta1pow[i] = beta1pow[i-1] * beta1
				beta2pow[i] = beta2pow[i-1] * beta2
			}
		}

		for i := 0; i < iterations; i++ {
			beta1pow[i] = 1 / (1 - beta1pow[i])
			beta2pow[i] = 1 / (1 - beta2pow[i])
		}

		return func(ctx context.Context, iteration int) {
			beta1powIterationLR := learningRate * beta1pow[iteration]
			beta2powIteration := beta2pow[iteration]

			commandBuffer := mps.CommandBufferFromContext(ctx)

			for i, node := range nodes {
				commandBuffer.UpdateWithAdam(
					node.Data,
					node.Grad,
					mm[i],
					vv[i],
					beta1,
					beta2,
					beta1powIterationLR,
					beta2powIteration,
				)
			}
		}
	}
}
