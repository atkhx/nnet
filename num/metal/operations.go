package metal

import (
	"context"
	"math"
	"math/rand"

	blas "github.com/atkhx/nnet/veclib/blas32"
	vdsp "github.com/atkhx/nnet/veclib/vdsp32"
)

func (aData *Data) Sqrt() *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for i, x := range aData.Data {
			output.Data[i] = float32(math.Sqrt(float64(x)))
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * 0.5 / y
		}
	}
	return output
}

func (aData *Data) Mean() *Data {
	output := New(NewDims(), aData)
	output.calcData = func(ctx context.Context) {
		output.Data[0] = aData.Data.Mean()
	}
	output.calcGrad = func(ctx context.Context) {
		aData.Grad.AddScalar(output.Grad[0] * 1.0 / float32(len(aData.Data)))
	}
	return output
}

func (aData *Data) MeanByRows() *Data {
	chunkSize := aData.Dims.W
	chunksCount := len(aData.Data) / chunkSize

	k := 1.0 / float32(chunkSize)

	output := New(NewDims(1, aData.Dims.H, aData.Dims.D), aData)
	output.calcData = func(ctx context.Context) {
		for i := 0; i < chunksCount; i++ {
			output.Data[i] = aData.Data[i*chunkSize : (i+1)*chunkSize].Mean()
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i := 0; i < chunksCount; i++ {
			aData.Grad[i*chunkSize : (i+1)*chunkSize].AddScalar(output.Grad[i] * k)
		}
	}
	return output
}

func (aData *Data) VarianceByRows(mean *Data) *Data {
	chunkSize := aData.Dims.W
	k := 1.0 / float32(chunkSize-1)

	output := New(NewDims(1, aData.Dims.H, aData.Dims.D), aData, mean)
	output.calcData = func(ctx context.Context) {
		for i := 0; i < len(output.Data); i++ {
			V := float32(0.0)
			M := mean.Data[i]
			for _, v := range aData.Data[i*chunkSize : (i+1)*chunkSize] {
				V += (v - M) * (v - M)
			}
			output.Data[i] = k * V
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, G := range output.Grad {
			M := mean.Data[i]
			for j, v := range aData.Data[i*chunkSize : (i+1)*chunkSize] {
				aData.Grad[i*chunkSize+j] += G * 2.0 * (v - M) * k
			}
		}
	}
	return output
}

func (aData *Data) Sigmoid() *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for i, x := range aData.Data {
			output.Data[i] = 1.0 / (1.0 + float32(math.Exp(float64(-x))))
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * y * (1 - y)
		}
	}
	return output
}

func (aData *Data) Tanh() *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for i, x := range aData.Data {
			output.Data[i] = float32(math.Tanh(float64(x)))
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * (1 - y*y)
		}
	}
	return output
}

func (aData *Data) Relu() *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for i, x := range aData.Data {
			if x > 0 {
				output.Data[i] = x
			} else {
				output.Data[i] = 0
			}
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, y := range output.Data {
			if y > 0 {
				aData.Grad[i] += output.Grad[i]
			}
		}
	}
	return output
}

func (aData *Data) AddScalar(k float32) *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for i, f := range aData.Data {
			output.Data[i] = f + k
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, g := range output.Grad {
			aData.Grad[i] += g
		}
	}
	return output
}

func (aData *Data) MulScalar(k float32) *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for i, f := range aData.Data {
			output.Data[i] = f * k
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, g := range output.Grad {
			aData.Grad[i] += g * k
		}
	}
	return output
}

func (aData *Data) Add(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] + bData.Data[bx]
		})
	}
	output.calcGrad = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset]
			bData.Grad[bx] += output.Grad[offset]
		})
	}
	return output
}

func (aData *Data) Sub(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] - bData.Data[bx]
		})
	}
	output.calcGrad = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset]
			bData.Grad[bx] -= output.Grad[offset]
		})
	}
	return output
}

func (aData *Data) Mul(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] * bData.Data[bx]
		})
	}
	output.calcGrad = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset] * bData.Data[bx]
			bData.Grad[bx] += output.Grad[offset] * aData.Data[ax]
		})
	}
	return output
}

func (aData *Data) Div(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	square := bData.Data.CopyZero()
	output.calcData = func(ctx context.Context) {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] / bData.Data[bx]
		})
	}
	output.calcGrad = func(ctx context.Context) {
		for k, v := range bData.Data {
			square[k] = float32(-math.Pow(float64(v), -2.0))
		}
		config.BroadCast(func(ax, bx, offset int) {
			gV := output.Grad[offset]
			if gV == 0 {
				return
			}

			if bV := bData.Data[bx]; bV != 0 {
				aData.Grad[ax] += gV / bV
			}

			if iV := aData.Data[ax]; iV != 0 {
				bData.Grad[bx] += gV * iV * square[bx]
			}
		})
	}
	return output
}

func (aData *Data) Softmax() *Data {
	chunkSize := aData.Dims.W

	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		output.Data.CopyFrom(aData.Data)
		for i := 0; i < len(output.Data); i += chunkSize {
			output.Data[i : i+chunkSize].Softmax()
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for b := 0; b < len(output.Data); b += chunkSize {
			oGrad := output.Grad[b : b+chunkSize]
			aGrad := aData.Grad[b : b+chunkSize]
			softmax := output.Data[b : b+chunkSize]

			s := float32(0.0)
			for i, softmaxI := range softmax {
				g := softmaxI * oGrad[i]
				s += g
				aGrad[i] += g
			}

			for i, softmaxI := range softmax {
				aGrad[i] -= softmaxI * s
			}
		}
	}
	return output
}

func (aData *Data) ConcatRows(bData ...*Data) *Data {
	if aData.Dims.H != bData[0].Dims.H {
		panic("height dimension must be equals")
	}

	if aData.Dims.D != bData[0].Dims.D {
		panic("depth dimension must be equals")
	}

	srcData := make([][]float32, 0, len(bData)+1)
	srcGrad := make([][]float32, 0, len(bData)+1)

	srcData = append(srcData, aData.Data)
	srcGrad = append(srcGrad, aData.Grad)

	for _, b := range bData {
		srcData = append(srcData, b.Data)
		srcGrad = append(srcGrad, b.Grad)
	}

	output := New(
		NewDims(aData.Dims.W*len(srcData), aData.Dims.H, aData.Dims.D),
		append(Nodes{aData}, bData...)...,
	)

	rowsCount := aData.Dims.H * aData.Dims.D
	colWidth := aData.Dims.W

	output.calcData = func(ctx context.Context) {
		var oOffset, bOffset int
		for i := 0; i < rowsCount; i++ {
			for _, nodeData := range srcData {
				copy(output.Data[oOffset:oOffset+colWidth], nodeData[bOffset:bOffset+colWidth])
				oOffset += colWidth
			}
			bOffset += colWidth
		}
	}

	output.calcGrad = func(ctx context.Context) {
		var oOffset, bOffset int
		for i := 0; i < rowsCount; i++ {
			for _, nodeGrad := range srcGrad {
				Float32s(nodeGrad[bOffset : bOffset+colWidth]).Add(output.Grad[oOffset : oOffset+colWidth])
				oOffset += colWidth
			}
			bOffset += colWidth
		}
	}
	return output
}

func (aData *Data) Dropout(prob float32) *Data {
	output := *aData
	output.Dims = aData.Dims
	output.srcNodes = Nodes{aData}
	output.skipResetGrad = true

	mask10 := New(aData.Dims)
	output.calcData = func(ctx context.Context) {
		for i, v := range aData.Data {
			if rand.Float32() > prob {
				output.Data[i] = v
				mask10.Data[i] = 1
			} else {
				mask10.Data[i] = 0
			}
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i := range aData.Grad {
			if mask10.Data[i] == 0 {
				aData.Grad[i] = 0
			}
		}
	}
	return &output
}

func (aData *Data) Reshape(dims Dims) *Data {
	if aData.Dims.Size() != dims.Size() {
		panic("total dimension size must be equal with original")
	}

	output := *aData
	output.Dims = dims
	output.srcNodes = Nodes{aData}
	output.skipResetGrad = true
	output.calcData = func(_ context.Context) {}
	output.calcGrad = func(_ context.Context) {}
	return &output
}

func (aData *Data) CrossEntropyPos(targets *Data) *Data {
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
	softmax := New(aData.Dims)
	output := New(targets.Dims, aData)
	output.calcData = func(ctx context.Context) {
		softmax.Data.CopyFrom(aData.Data)
		for i := 0; i < len(softmax.Data); i += chunkSize {
			softmax.Data[i : i+chunkSize].Softmax()
		}

		for rowIdx, correctIdx := range targets.Data {
			output.Data[rowIdx] = float32(-math.Log(float64(softmax.Data[rowIdx*chunkSize+int(correctIdx)])))
		}
	}
	output.calcGrad = func(ctx context.Context) {
		offset := 0
		for rowIdx, ci := range targets.Data {
			correctIdx := int(ci)

			oGrad := output.Grad[rowIdx]
			aGrad := aData.Grad[offset : offset+chunkSize]
			softmax := softmax.Data[offset : offset+chunkSize]
			offset += chunkSize

			for i, softmaxJ := range softmax {
				if i == correctIdx {
					aGrad[i] += oGrad * (softmaxJ - 1)
				} else {
					aGrad[i] += oGrad * softmaxJ
				}
			}
		}
	}
	return output
}

func (aData *Data) Embeddings(tEmbeddings, pEmbeddings *Data) *Data {
	if tEmbeddings.Dims.W != pEmbeddings.Dims.W {
		panic("features count must be equal")
	}

	featuresCount := tEmbeddings.Dims.W

	contextSize := aData.Dims.W
	tokensCount := aData.Dims.H

	output := New(NewDims(featuresCount, contextSize, tokensCount), tEmbeddings)
	output.calcData = func(ctx context.Context) {
		p := 0
		for i, s := range aData.Data.ToInt() {
			tFeatures := tEmbeddings.Data[s*featuresCount : (s+1)*featuresCount]
			pFeatures := pEmbeddings.Data[p*featuresCount : (p+1)*featuresCount]

			outBuffer := output.Data[i*featuresCount : (i+1)*featuresCount]
			outBuffer.CopyFrom(tFeatures)
			outBuffer.Add(pFeatures)

			p++
			if p == contextSize {
				p = 0
			}
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for i, s := range aData.Data.ToInt() {
			tGrads := tEmbeddings.Grad[s*featuresCount : (s+1)*featuresCount]
			tGrads.Add(output.Grad[i*featuresCount : (i+1)*featuresCount])
		}
	}
	return output
}

func (aData *Data) Transpose() *Data {
	IW, IH := aData.Dims.W, aData.Dims.H
	WH := aData.Dims.W * aData.Dims.H

	mGradBuffer := NewFloat32s(WH)

	output := aData.NewLinkedCopy()
	output.Dims.W = aData.Dims.H
	output.Dims.H = aData.Dims.W

	output.calcData = func(ctx context.Context) {
		for offset := 0; offset < len(aData.Data); offset += WH {
			vdsp.MtransD(aData.Data[offset:offset+WH], 1, output.Data[offset:offset+WH], 1, IW, IH)
		}
	}

	output.calcGrad = func(ctx context.Context) {
		for offset := 0; offset < len(aData.Grad); offset += WH {
			vdsp.MtransD(output.Grad[offset:offset+WH], 1, mGradBuffer, 1, IH, IW)
			aData.Grad[offset : offset+WH].Add(mGradBuffer)
		}
	}
	return output
}

func (aData *Data) TriangleLowerSoftmax() *Data {
	W, H, D := aData.Dims.W, aData.Dims.H, aData.Dims.D
	WH := W * H

	output := aData.NewLinkedCopy()
	output.calcData = func(ctx context.Context) {
		for z := 0; z < D; z++ {
			for y := 0; y < H; y++ {
				c := z*WH + y*W
				aData.Data[c : c+y+1].SoftmaxTo(output.Data[c : c+y+1])
			}
		}
	}
	output.calcGrad = func(ctx context.Context) {
		for z := 0; z < D; z++ {
			for y := 0; y < H; y++ {
				c := z*WH + y*W

				iGrad := aData.Grad[c : c+y+1]
				oGrad := output.Grad[c : c+y+1]

				softmax := output.Data[c : c+y+1]

				g := float32(0.0)
				s := float32(0.0)
				for i, softmaxI := range softmax {
					g = softmaxI * oGrad[i]
					s += g
					iGrad[i] += g
				}

				for i, softmaxI := range softmax {
					iGrad[i] -= softmaxI * s
				}
			}
		}
	}
	return output
}

type mmConfig struct {
	alpha float32
}

type mmOption func(*mmConfig)

func WithMatrixMultiplyAlpha(alpha float32) mmOption {
	return func(config *mmConfig) {
		config.alpha = alpha
	}
}

func (aData *Data) MatrixMultiply2D(bData *Data, options ...mmOption) *Data {
	if aData.Dims.W != bData.Dims.H {
		panic("aData width must be equal bData height")
	}

	if bData.Dims.D != 1 || aData.Dims.D != 1 {
		panic("matrix is not 2D")
	}

	cfg := &mmConfig{alpha: 1.0}
	for _, option := range options {
		option(cfg)
	}
	alpha := cfg.alpha

	oH := aData.Dims.H
	oW := bData.Dims.W
	oD := aData.Dims.D
	aW := aData.Dims.W

	output := New(Dims{W: oW, H: oH, D: oD}, aData, bData)
	output.calcData = func(ctx context.Context) {
		blas.MatrixMultiplyAB(aW, aData.Data, bData.Data, output.Data, alpha, 0.0)
	}
	output.calcGrad = func(ctx context.Context) {
		blas.MatrixMultiplyAonTransposedB(oW, output.Grad, bData.Data, aData.Grad, alpha, 1)
		blas.MatrixMultiplyTransposedAonB(oH, aData.Data, output.Grad, bData.Grad, alpha, 1)
	}
	return output
}

func (aData *Data) MatrixMultiply(bData *Data, options ...mmOption) *Data {
	if aData.Dims.W != bData.Dims.H {
		panic("aData width must be equal bData height")
	}

	if aData.Dims.D == 1 && bData.Dims.D == 1 {
		return aData.MatrixMultiply2D(bData, options...)
	}

	cfg := &mmConfig{alpha: 1.0}
	for _, option := range options {
		option(cfg)
	}
	alpha := cfg.alpha

	oD := aData.Dims.D
	if bData.Dims.D > oD {
		oD = bData.Dims.D
	}

	oW := bData.Dims.W
	oH := aData.Dims.H

	output := New(Dims{W: oW, H: oH, D: oD}, aData, bData)

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

	output.calcData = func(ctx context.Context) {
		var ozOffset, izOffset, fzOffset int
		for z := 0; z < oD; z++ {
			blas.MatrixMultiplyAB(aData.Dims.W,
				aData.Data[izOffset:izOffset+iWH],
				bData.Data[fzOffset:fzOffset+fWH],
				output.Data[ozOffset:ozOffset+oWH], alpha, 0)

			ozOffset += oWH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.calcGrad = func(ctx context.Context) {
		var ozOffset, izOffset, fzOffset int
		for z := 0; z < oD; z++ {
			blas.MatrixMultiplyAonTransposedB(oW,
				output.Grad[ozOffset:ozOffset+oWH],
				bData.Data[fzOffset:fzOffset+fWH],
				aData.Grad[izOffset:izOffset+iWH],
				alpha, 1)

			blas.MatrixMultiplyTransposedAonB(aData.Dims.H,
				aData.Data[izOffset:izOffset+iWH],
				output.Grad[ozOffset:ozOffset+oWH],
				bData.Grad[fzOffset:fzOffset+fWH],
				alpha, 1)

			ozOffset += oWH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	return output
}
