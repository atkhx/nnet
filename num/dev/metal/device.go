package metal

import (
	"context"
	"math"

	"github.com/atkhx/mps"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/num/broadcast"
)

type Device struct {
	dev *mps.MTLDevice
}

func NewDevice() *Device {
	return &Device{
		dev: mps.NewMTLDevice(),
	}
}

func (d *Device) Close() error {
	d.dev.Release()
	return nil
}

func (d *Device) CreateCommandQueue() *mps.MTLCommandQueue {
	return d.dev.CreateCommandQueue()
}

type dataOpts struct {
	dataBuffer *mps.MTLBuffer
	gradBuffer *mps.MTLBuffer
}

func (d *Device) NewData(dims num.Dims, srcNodes ...*num.Data) *num.Data {
	dataBuffer := d.dev.CreateNewBufferWithLength(dims.Size())
	gradBuffer := d.dev.CreateNewBufferWithLength(dims.Size())

	return num.NewData(
		dataBuffer.GetData(),
		gradBuffer.GetData(),
		dims,
		dataOpts{
			dataBuffer: dataBuffer,
			gradBuffer: gradBuffer,
		},
		srcNodes...,
	)
}

func (d *Device) newLinkedCopy(aData *num.Data) *num.Data {
	return d.NewData(aData.Dims, aData)
}

func (d *Device) NewDataRandNormWeighted(dims num.Dims, w float32) *num.Data {
	data := d.NewData(dims)
	Float32s(data.Data).RandNormWeighted(w)
	return data
}

func (d *Device) NewTokenEmbeddingTable(featuresCount, alphabetSize int) *num.Data {
	data := d.NewData(num.NewDims(featuresCount, alphabetSize))
	Float32s(data.Data).RandNorm()
	return data
}

func (d *Device) NewPositionEmbeddingTable(featuresCount, contextSize int) *num.Data {
	result := d.NewData(num.NewDims(featuresCount, contextSize))

	k := 0
	for j := 0; j < contextSize; j++ {
		for i := 0; i < featuresCount; i++ {
			if i%2 == 0 {
				result.Data[k] = float32(math.Sin(float64(k) / math.Pow(10_000, float64(i+1)/float64(featuresCount))))
			} else {
				result.Data[k] = float32(math.Cos(float64(k) / math.Pow(10_000, float64(i+1)/float64(featuresCount))))
			}
			k++
		}
	}
	return result
}

func (d *Device) FillDataWithZeros(aData *num.Data) {
	Float32s(aData.Data).Zero()
}

func (d *Device) FillDataWithOnes(aData *num.Data) {
	Float32s(aData.Data).Ones()
}

func (d *Device) FillGradWithZeros(aData *num.Data) {
	Float32s(aData.Grad).Zero()
}

func (d *Device) FillGradWithOnes(aData *num.Data) {
	Float32s(aData.Grad).Ones()
}

func (d *Device) GetDataDims(aData *num.Data) num.Dims {
	return aData.Dims
}

func (d *Device) GetDataLength(aData *num.Data) int {
	return len(aData.Data)
}

func (d *Device) Sqrt(aData *num.Data) *num.Data {
	output := d.newLinkedCopy(aData)
	output.CalcData = func(ctx context.Context) {
		Float32s(aData.Data).SqrtTo(output.Data)
	}
	output.CalcGrad = func(ctx context.Context) {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * 0.5 / y
		}
	}
	return output
}

func (d *Device) Mean(aData *num.Data) *num.Data {
	output := d.NewData(num.NewDims(), aData)
	output.CalcData = func(ctx context.Context) {
		output.Data[0] = Float32s(aData.Data).Mean()
	}
	output.CalcGrad = func(ctx context.Context) {
		Float32s(aData.Grad).AddScalar(output.Grad[0] * 1.0 / float32(len(aData.Data)))
		//Float32s(aData.Grad).AddScalar(output.Grad[0])
	}
	return output
}

func (d *Device) MeanByRows(aData *num.Data) *num.Data {
	chunkSize := aData.Dims.W
	chunksCount := len(aData.Data) / chunkSize

	k := 1.0 / float32(chunkSize)

	output := d.NewData(num.NewDims(1, aData.Dims.H, aData.Dims.D), aData)
	output.CalcData = func(ctx context.Context) {
		for i := 0; i < chunksCount; i++ {
			output.Data[i] = Float32s(aData.Data[i*chunkSize : (i+1)*chunkSize]).Mean()
		}
	}
	output.CalcGrad = func(ctx context.Context) {
		for i := 0; i < chunksCount; i++ {
			Float32s(aData.Grad[i*chunkSize : (i+1)*chunkSize]).AddScalar(output.Grad[i] * k)
		}
	}
	return output
}

func (d *Device) VarianceByRows(aData, mean *num.Data) *num.Data {
	chunkSize := aData.Dims.W
	chunksCount := len(aData.Data) / chunkSize

	k := 1.0 / float32(chunkSize-1)

	output := d.NewData(num.NewDims(1, aData.Dims.H, aData.Dims.D), aData, mean)
	output.CalcData = func(ctx context.Context) {
		for i := 0; i < chunksCount; i++ {
			output.Data[i] = Float32s(aData.Data[i*chunkSize : (i+1)*chunkSize]).Variance(mean.Data[i])
		}
	}
	output.CalcGrad = func(ctx context.Context) {
		for i, g := range output.Grad {
			for j, v := range aData.Data[i*chunkSize : (i+1)*chunkSize] {
				aData.Grad[i*chunkSize+j] += g * 2.0 * k * (v - mean.Data[i])
			}
		}
	}
	return output
}

func (d *Device) LNorm(aData, gamma, beta *num.Data) *num.Data {
	eps := float32(0.000000001)

	mean := d.MeanByRows(aData)           // matrix [1, H, D]
	vars := d.VarianceByRows(aData, mean) // matrix [1, H, D]
	xSub := d.Sub(aData, mean)            // matrix [W, H, D]
	sqrt := d.Sqrt(vars)                  // matrix [1, H, D] - √var
	sqrtEps := d.AddScalar(sqrt, eps)     // matrix [1, H, D] - √var - eps
	xDiv := d.Div(xSub, sqrtEps)          // matrix [W, H, D] - aData[i] /= sqrtEps[ColI]
	xMul := d.Mul(gamma, xDiv)            // matrix [W, H, D] - aData[i] *= gamma[ColI]
	return d.Add(xMul, beta)              // matrix [W, H, D] - aData[i] += beta[ColI]
}

func (d *Device) Relu(aData *num.Data) *num.Data {
	output := d.newLinkedCopy(aData)

	aDataBuffer := aData.Opts.(dataOpts).dataBuffer
	aGradBuffer := aData.Opts.(dataOpts).gradBuffer
	oDataBuffer := output.Opts.(dataOpts).dataBuffer
	oGradBuffer := output.Opts.(dataOpts).gradBuffer

	output.CalcData = func(ctx context.Context) {
		cbuf := mps.CommandBufferFromContext(ctx)
		cbuf.ReLuMTLBuffer(oDataBuffer, aDataBuffer)
	}

	output.CalcGrad = func(ctx context.Context) {
		cbuf := mps.CommandBufferFromContext(ctx)
		cbuf.ReLuMTLBufferBwd(aGradBuffer, oGradBuffer, oDataBuffer)
	}
	return output
}

func (d *Device) AddScalar(aData *num.Data, k float32) *num.Data {
	output := d.newLinkedCopy(aData)
	output.CalcData = func(ctx context.Context) {
		Float32s(aData.Data).AddScalarTo(output.Data, k)
	}
	output.CalcGrad = func(ctx context.Context) {
		Float32s(aData.Grad).Add(output.Grad)
	}
	return output
}

func (d *Device) MulScalar(aData *num.Data, k float32) *num.Data {
	output := d.newLinkedCopy(aData)
	output.CalcData = func(ctx context.Context) {
		Float32s(aData.Data).MulScalarTo(output.Data, k)
	}
	output.CalcGrad = func(ctx context.Context) {
		Float32s(aData.Grad).AddWeighted(output.Grad, k)
	}
	return output
}

func (d *Device) Add(aData, bData *num.Data) *num.Data {
	config := broadcast.NewConfig(aData.Dims, bData.Dims)
	output := d.NewData(config.OutDims, aData, bData)

	addRep := func(WH, aSize int) *num.Data {
		output.CalcData = func(ctx context.Context) {
			cbuf := mps.CommandBufferFromContext(ctx)
			cbuf.Copy(output.Opts.(dataOpts).dataBuffer, aData.Opts.(dataOpts).dataBuffer, 0, 0, aSize)

			for i := 0; i < aSize; i += WH {
				cbuf.Add(output.Opts.(dataOpts).dataBuffer, bData.Opts.(dataOpts).dataBuffer, i, 0, WH)
			}
		}
		output.CalcGrad = func(ctx context.Context) {
			cbuf := mps.CommandBufferFromContext(ctx)
			cbuf.Add(aData.Opts.(dataOpts).gradBuffer, output.Opts.(dataOpts).gradBuffer, 0, 0, aSize)

			for i := 0; i < aSize; i += WH {
				cbuf.Add(bData.Opts.(dataOpts).gradBuffer, output.Opts.(dataOpts).gradBuffer, 0, i, WH)
			}
		}
		return output
	}

	if aData.Dims == bData.Dims {
		output.CalcData = func(ctx context.Context) {
			cbuf := mps.CommandBufferFromContext(ctx)
			cbuf.AddTo(
				output.Opts.(dataOpts).dataBuffer,
				aData.Opts.(dataOpts).dataBuffer,
				bData.Opts.(dataOpts).dataBuffer,
			)
		}
		output.CalcGrad = func(ctx context.Context) {
			cbuf := mps.CommandBufferFromContext(ctx)
			cbuf.Add(aData.Opts.(dataOpts).gradBuffer, output.Opts.(dataOpts).gradBuffer, 0, 0, len(aData.Grad))
			cbuf.Add(bData.Opts.(dataOpts).gradBuffer, output.Opts.(dataOpts).gradBuffer, 0, 0, len(aData.Grad))
		}
		return output
	}

	if aData.Dims.W == bData.Dims.W && aData.Dims.H == bData.Dims.H && bData.Dims.D == 1 {
		return addRep(aData.Dims.W*aData.Dims.H, aData.Dims.Size())
	}

	if aData.Dims.W == bData.Dims.W && bData.Dims.H == 1 && bData.Dims.D == 1 {
		return addRep(aData.Dims.W, aData.Dims.Size())
	}

	output.CalcData = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] + bData.Data[bx]
		})
	}
	output.CalcGrad = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset]
			bData.Grad[bx] += output.Grad[offset]
		})
	}
	return output
}

func (d *Device) Sub(aData, bData *num.Data) *num.Data {
	config := broadcast.NewConfig(aData.Dims, bData.Dims)
	output := d.NewData(config.OutDims, aData, bData)
	output.CalcData = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] - bData.Data[bx]
		})
	}
	output.CalcGrad = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset]
			bData.Grad[bx] -= output.Grad[offset]
		})
	}
	return output
}

func (d *Device) Mul(aData, bData *num.Data) *num.Data {
	config := broadcast.NewConfig(aData.Dims, bData.Dims)
	output := d.NewData(config.OutDims, aData, bData)
	output.CalcData = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] * bData.Data[bx]
		})
	}
	output.CalcGrad = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset] * bData.Data[bx]
			bData.Grad[bx] += output.Grad[offset] * aData.Data[ax]
		})
	}
	return output
}

func (d *Device) Div(aData, bData *num.Data) *num.Data {
	config := broadcast.NewConfig(aData.Dims, bData.Dims)
	output := d.NewData(config.OutDims, aData, bData)
	output.CalcData = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] / bData.Data[bx]
		})
	}
	output.CalcGrad = func(ctx context.Context) {
		config.Broadcast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset] / bData.Data[bx]
			bData.Grad[bx] -= output.Grad[offset] * output.Data[ax] / bData.Data[bx]
		})
	}
	return output
}

func (d *Device) ConcatByRows(bData ...*num.Data) *num.Data {
	dims := bData[0].Dims

	output := d.NewData(num.NewDims(dims.W*len(bData), dims.H, dims.D), bData...)
	output.CalcData = func(ctx context.Context) {
		var oOffset, bOffset int
		for i := 0; i < dims.H*dims.D; i++ {
			for _, node := range bData {
				Float32s(output.Data[oOffset : oOffset+dims.W]).CopyFrom(node.Data[bOffset : bOffset+dims.W])
				oOffset += dims.W
			}
			bOffset += dims.W
		}
	}
	output.CalcGrad = func(ctx context.Context) {
		var oOffset, bOffset int
		for i := 0; i < dims.H*dims.D; i++ {
			for _, node := range bData {
				Float32s(node.Grad[bOffset : bOffset+dims.W]).Add(output.Grad[oOffset : oOffset+dims.W])
				oOffset += dims.W
			}
			bOffset += dims.W
		}
	}
	return output
}

func (d *Device) Dropout(aData *num.Data, prob float32) *num.Data {
	if prob == 0 {
		return aData
	}

	//mask10 := make([]bool, aData.Dims.Size())
	output := d.newLinkedCopy(aData)

	mask := d.NewData(aData.Dims)

	output.CalcData = func(ctx context.Context) {
		cbuf := mps.CommandBufferFromContext(ctx)
		cbuf.DropoutBuffer(
			output.Opts.(dataOpts).dataBuffer,
			aData.Opts.(dataOpts).dataBuffer,
			mask.Opts.(dataOpts).dataBuffer,
			prob,
		)
	}
	output.CalcGrad = func(ctx context.Context) {
		for i, g := range output.Grad {
			if mask.Data[i] > 0 {
				aData.Grad[i] += g
			}
		}
	}
	//output.CalcData = func(ctx context.Context) {
	//	d.FillDataWithZeros(output)
	//	for i, v := range aData.Data {
	//		if mask10[i] = randGenerator.Float32() > prob; mask10[i] {
	//			output.Data[i] = v
	//		}
	//	}
	//}
	//output.CalcGrad = func(ctx context.Context) {
	//	for i, g := range output.Grad {
	//		if mask10[i] {
	//			aData.Grad[i] += g
	//		}
	//	}
	//}
	return output
}

func (d *Device) Reshape(aData *num.Data, dims num.Dims) *num.Data {
	if aData.Dims.Size() != dims.Size() {
		panic("total dimension size must be equal with original")
	}

	output := *aData
	output.Dims = dims
	output.SrcNodes = num.Nodes{aData}
	output.SkipResetGrad = true
	output.CalcData = func(_ context.Context) {}
	output.CalcGrad = func(_ context.Context) {}
	return &output
}

func (d *Device) CrossEntropyPos(aData, targets *num.Data) *num.Data {
	if targets.Dims.W != 1 {
		panic("target width must be equal 1")
	}

	if targets.Dims.H != aData.Dims.H {
		panic("target height must be equal aData height")
	}

	if targets.Dims.D != aData.Dims.D {
		panic("target depth must be equal aData depth")
	}

	chunkSize := aData.Dims.W
	softmax := Float32s(aData.Data).CopyZero()

	output := d.NewData(targets.Dims, aData)
	output.CalcData = func(ctx context.Context) {
		softmax.CopyFrom(aData.Data)
		for i := 0; i < len(softmax); i += chunkSize {
			softmax[i : i+chunkSize].Softmax()
		}

		for rowIdx, correctIdx := range targets.Data {
			output.Data[rowIdx] = float32(-math.Log(float64(softmax[rowIdx*chunkSize+int(correctIdx)])))
		}
	}
	output.CalcGrad = func(ctx context.Context) {
		offset := 0
		for rowIdx, ci := range targets.Data {
			correctIdx := int(ci)
			for i, softmaxI := range softmax[offset : offset+chunkSize] {
				if i == correctIdx {
					aData.Grad[offset+i] += output.Grad[rowIdx] * (softmaxI - 1)
				} else {
					aData.Grad[offset+i] += output.Grad[rowIdx] * softmaxI
				}
			}
			offset += chunkSize
		}
	}
	return output
}

func (d *Device) Embeddings(aData *num.Data, tEmbeddings, pEmbeddings *num.Data) *num.Data {
	if tEmbeddings.Dims.W != pEmbeddings.Dims.W {
		panic("features count must be equal")
	}

	featuresCount := tEmbeddings.Dims.W

	contextSize := aData.Dims.W
	tokensCount := aData.Dims.H

	output := d.NewData(num.NewDims(featuresCount, contextSize, tokensCount), tEmbeddings)
	output.CalcData = func(ctx context.Context) {
		p := 0
		for i, s := range Float32s(aData.Data).ToInt() {
			tFeatures := tEmbeddings.Data[s*featuresCount : (s+1)*featuresCount]
			pFeatures := pEmbeddings.Data[p*featuresCount : (p+1)*featuresCount]
			outBuffer := output.Data[i*featuresCount : (i+1)*featuresCount]

			Float32s(tFeatures).AddTo(outBuffer, pFeatures)

			p++
			if p == contextSize {
				p = 0
			}
		}
	}
	output.CalcGrad = func(ctx context.Context) {
		for i, s := range Float32s(aData.Data).ToInt() {
			tGrads := Float32s(tEmbeddings.Grad[s*featuresCount : (s+1)*featuresCount])
			tGrads.Add(output.Grad[i*featuresCount : (i+1)*featuresCount])
		}
	}
	return output
}

func (d *Device) Transpose(aData *num.Data) *num.Data {
	IW, IH := aData.Dims.W, aData.Dims.H

	output := d.newLinkedCopy(aData)
	output.Dims.W = aData.Dims.H
	output.Dims.H = aData.Dims.W

	output.CalcData = func(ctx context.Context) {
		Float32s(aData.Data).TransposeTo(output.Data, IW, IH)
	}

	output.CalcGrad = func(ctx context.Context) {
		Float32s(output.Grad).TransposeAndAddTo(aData.Grad, IH, IW)
	}
	return output
}

func (d *Device) TriangleLowerSoftmax(aData *num.Data) *num.Data {
	W, H, D, WH := aData.Dims.W, aData.Dims.H, aData.Dims.D, aData.Dims.W*aData.Dims.H

	output := d.newLinkedCopy(aData)
	output.CalcData = func(ctx context.Context) {
		for z := 0; z < D; z++ {
			for y := 0; y < H; y++ {
				c := z*WH + y*W
				Float32s(aData.Data[c : c+y+1]).SoftmaxTo(output.Data[c : c+y+1])
			}
		}
	}
	output.CalcGrad = func(ctx context.Context) {
		for z := 0; z < D; z++ {
			for y := 0; y < H; y++ {
				c := z*WH + y*W

				iGrad := aData.Grad[c : c+y+1]
				oGrad := output.Grad[c : c+y+1]
				softmax := output.Data[c : c+y+1]

				var softmaxByGrads float32
				for i, softmaxI := range softmax {
					softmaxByGradI := softmaxI * oGrad[i]
					softmaxByGrads += softmaxByGradI

					iGrad[i] += softmaxByGradI
				}

				for i, softmaxI := range softmax {
					iGrad[i] -= softmaxI * softmaxByGrads
				}
			}
		}
	}
	return output
}

func (d *Device) MatrixMultiply2D(aData, bData *num.Data, alpha float32) *num.Data {
	if aData.Dims.W != bData.Dims.H {
		panic("aData width must be equal bData height")
	}

	if bData.Dims.D != 1 || aData.Dims.D != 1 {
		panic("matrix is not 2D")
	}

	oH := aData.Dims.H
	oW := bData.Dims.W
	oD := aData.Dims.D

	output := d.NewData(num.Dims{W: oW, H: oH, D: oD}, aData, bData)

	aDataM := aData.Opts.(dataOpts).dataBuffer.CreateMatrix(aData.Dims.W, aData.Dims.H, 0)
	bDataM := bData.Opts.(dataOpts).dataBuffer.CreateMatrix(bData.Dims.W, bData.Dims.H, 0)
	cDataM := output.Opts.(dataOpts).dataBuffer.CreateMatrix(oW, oH, 0)

	output.CalcData = func(ctx context.Context) {
		cbuf := mps.CommandBufferFromContext(ctx)
		cbuf.MatrixMultiplyAB(aDataM, bDataM, cDataM, alpha, 0.0)
	}

	aGradM := aData.Opts.(dataOpts).gradBuffer.CreateMatrix(aData.Dims.W, aData.Dims.H, 0)
	bGradM := bData.Opts.(dataOpts).gradBuffer.CreateMatrix(bData.Dims.W, bData.Dims.H, 0)
	cGradM := output.Opts.(dataOpts).gradBuffer.CreateMatrix(oW, oH, 0)

	output.CalcGrad = func(ctx context.Context) {
		cbuf := mps.CommandBufferFromContext(ctx)
		cbuf.MatrixMultiplyATB(cGradM, bDataM, aGradM, alpha, 1.0)
		cbuf.MatrixMultiplyTAB(aDataM, cGradM, bGradM, alpha, 1.0)
	}
	return output
}

func (d *Device) MatrixMultiply3D(aData, bData *num.Data, alpha float32) *num.Data {
	if aData.Dims.W != bData.Dims.H {
		panic("aData width must be equal bData height")
	}

	if aData.Dims.D == 1 && bData.Dims.D == 1 {
		return d.MatrixMultiply2D(aData, bData, alpha)
	}

	oD := aData.Dims.D
	if bData.Dims.D > oD {
		oD = bData.Dims.D
	}

	oW := bData.Dims.W
	oH := aData.Dims.H

	output := d.NewData(num.Dims{W: oW, H: oH, D: oD}, aData, bData)

	iWH := aData.Dims.W * aData.Dims.H
	fWH := bData.Dims.W * bData.Dims.H
	oWH := output.Dims.W * output.Dims.H

	izStep := 1
	fzStep := 1

	if aData.Dims.D != bData.Dims.D {
		switch {
		case aData.Dims.D == 1:
			izStep = 0
		case bData.Dims.D == 1:
			fzStep = 0
		default:
			panic("aData's and bData's dept must be equal or one of them must be 1")
		}
	}

	if bData.Dims.D == 1 {
		// aData - 3D matrix
		// bData - 1D matrix
		// cData - 3D matrix

		aDataM := aData.Opts.(dataOpts).dataBuffer.CreateMatrix(aData.Dims.W, aData.Dims.H*aData.Dims.D, 0)
		bDataM := bData.Opts.(dataOpts).dataBuffer.CreateMatrix(bData.Dims.W, bData.Dims.H, 0)
		cDataM := output.Opts.(dataOpts).dataBuffer.CreateMatrix(oW, oH*oD, 0)

		output.CalcData = func(ctx context.Context) {
			cbuf := mps.CommandBufferFromContext(ctx)
			cbuf.MatrixMultiplyAB(aDataM, bDataM, cDataM, alpha, 0.0)
		}
	} else {
		type ABC struct {
			aDataM, bDataM, cDataM *mps.Matrix
		}

		list := make([]ABC, oD)
		{
			var ozOffset, izOffset, fzOffset int
			for z := 0; z < oD; z++ {
				list[z].aDataM = aData.Opts.(dataOpts).dataBuffer.CreateMatrix(aData.Dims.W, aData.Dims.H, izOffset)
				list[z].bDataM = bData.Opts.(dataOpts).dataBuffer.CreateMatrix(bData.Dims.W, bData.Dims.H, fzOffset)
				list[z].cDataM = output.Opts.(dataOpts).dataBuffer.CreateMatrix(oW, oH, ozOffset)

				ozOffset += oWH
				izOffset += izStep * iWH
				fzOffset += fzStep * fWH
			}
		}

		output.CalcData = func(ctx context.Context) {
			cbuf := mps.CommandBufferFromContext(ctx)
			for z := 0; z < oD; z++ {
				cbuf.MatrixMultiplyAB(list[z].aDataM, list[z].bDataM, list[z].cDataM, alpha, 0.0)
			}
		}
	}

	type ABCGrad struct {
		aDataM, bDataM, cDataM *mps.Matrix
		aGradM, bGradM, cGradM *mps.Matrix
	}

	list := make([]ABCGrad, oD)
	{
		var ozOffset, izOffset, fzOffset int
		for z := 0; z < oD; z++ {
			list[z].aDataM = aData.Opts.(dataOpts).dataBuffer.CreateMatrix(aData.Dims.W, aData.Dims.H, izOffset)
			list[z].bDataM = bData.Opts.(dataOpts).dataBuffer.CreateMatrix(bData.Dims.W, bData.Dims.H, fzOffset)
			list[z].cDataM = output.Opts.(dataOpts).dataBuffer.CreateMatrix(oW, oH, ozOffset)

			list[z].aGradM = aData.Opts.(dataOpts).gradBuffer.CreateMatrix(aData.Dims.W, aData.Dims.H, izOffset)
			list[z].bGradM = bData.Opts.(dataOpts).gradBuffer.CreateMatrix(bData.Dims.W, bData.Dims.H, fzOffset)
			list[z].cGradM = output.Opts.(dataOpts).gradBuffer.CreateMatrix(oW, oH, ozOffset)

			ozOffset += oWH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.CalcGrad = func(ctx context.Context) {
		cbuf := mps.CommandBufferFromContext(ctx)
		for z := 0; z < oD; z++ {
			cbuf.MatrixMultiplyATB(list[z].cGradM, list[z].bDataM, list[z].aGradM, alpha, 1.0)
			cbuf.MatrixMultiplyTAB(list[z].aDataM, list[z].cGradM, list[z].bGradM, alpha, 1.0)
		}
	}

	return output
}

func (d *Device) GetOptimizerAdam(iterations int, beta1, beta2, learningRate, eps float32) func(nodes []*num.Data) func(iteration int) {
	iterations++
	return func(nodes []*num.Data) func(iteration int) {

		mm := make([]*mps.MTLBuffer, len(nodes))
		vv := make([]*mps.MTLBuffer, len(nodes))

		for i, node := range nodes {
			mm[i] = d.dev.CreateNewBufferWithLength(len(node.Data))
			vv[i] = d.dev.CreateNewBufferWithLength(len(node.Data))
		}

		beta1pow := NewFloat32s(iterations)
		beta2pow := NewFloat32s(iterations)

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

		commandQueue := d.dev.CreateCommandQueue()

		return func(iteration int) {
			beta1powIterationLR := learningRate * beta1pow[iteration]
			beta2powIteration := beta2pow[iteration]

			commandBuffer := commandQueue.GetCommandBuffer()
			defer func() {
				commandBuffer.Wait()
				//commandBuffer.Release()
			}()

			for i, node := range nodes {
				commandBuffer.UpdateWithAdam(
					node.Opts.(dataOpts).dataBuffer,
					node.Opts.(dataOpts).gradBuffer,
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
