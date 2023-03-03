package data

import (
	"fmt"
	"math"
)

func NewMatrixRandom(colsCount, rowsCount, chanCount int) *Matrix {
	return NewMatrix(colsCount, rowsCount, chanCount, MakeRandom(colsCount*rowsCount*chanCount))
}

func NewMatrix(colsCount, rowsCount, chanCount int, data []float64) *Matrix {
	if len(data) != colsCount*rowsCount*chanCount {
		panic(fmt.Sprintf("invalid dimensions: data length %d, colsCount*rowsCount*chan %d", len(data), colsCount*rowsCount*chanCount))
	}

	return &Matrix{
		ColsCount: colsCount,
		RowsCount: rowsCount,
		ChanCount: chanCount,

		Data: data,
		Grad: make([]float64, len(data)),
	}
}

func NewMatrixResult(colsCount, rowsCount, chanCount int, data []float64, from *Source) (outMatrix *Matrix) {
	outMatrix = NewMatrix(colsCount, rowsCount, chanCount, data)
	outMatrix.From = from
	return
}

type Matrix struct {
	ColsCount int
	RowsCount int
	ChanCount int

	Data []float64
	Grad []float64
	From *Source

	backwardCalled bool
}

func (m *Matrix) GradsMatrix() *Matrix {
	return NewMatrix(m.ColsCount, m.RowsCount, m.ChanCount, m.Grad)
}

func (m *Matrix) gradRowFloats(rowIndex int) []float64 {
	colsCount := m.ColsCount
	rowOffset := rowIndex * colsCount

	return m.Grad[rowOffset : rowOffset+colsCount]
}

func (m *Matrix) col(colIndex, chanIndex int) []float64 {
	return MatrixColFloatsChan(
		m.ColsCount,
		m.RowsCount,
		colIndex,
		chanIndex,
		m.Data,
	)
}

func (m *Matrix) colGrads(colIndex, chanIndex int) []float64 {
	return MatrixColFloatsChan(
		m.ColsCount,
		m.RowsCount,
		colIndex,
		chanIndex,
		m.Grad,
	)
}

func (m *Matrix) row(rowIndex, chanIndex int) []float64 {
	return MatrixRowFloatsChan(
		m.ColsCount,
		m.RowsCount,
		rowIndex,
		chanIndex,
		m.Data,
	)
}

func (m *Matrix) rowGrads(rowIndex, chanIndex int) []float64 {
	return MatrixRowFloatsChan(
		m.ColsCount,
		m.RowsCount,
		rowIndex,
		chanIndex,
		m.Grad,
	)
}

func (m *Matrix) Transpose() (outMatrix *Matrix) {
	if m.ChanCount > 1 {
		panic("transpose not implemented for multy-channel matrix")
	}

	r, c, d := MatrixTranspose(m.ColsCount, m.RowsCount, m.Data)

	outMatrix = NewMatrix(r, c, m.ChanCount, d)
	outMatrix.From = NewSource(func() {
		AddTo(m.Grad, NewMatrix(
			outMatrix.ColsCount,
			outMatrix.RowsCount,
			outMatrix.ChanCount,
			outMatrix.Grad,
		).Transpose().Data)
	}, m)
	return
}

func (m *Matrix) Flat() (outMatrix *Matrix) {
	return NewMatrixResult(len(m.Data), 1, 1, CopyWithData(m.Data), NewSource(func() {
		AddTo(m.Grad, outMatrix.Grad)
	}, m))
}

func (m *Matrix) Conv(
	imageWidth, imageHeight, filterSize, padding, stride int,
	filters *Matrix,
	biases *Matrix,
) (outMatrix *Matrix) {
	imagesCount := m.ChanCount
	filtersCount := filters.ChanCount
	channels := m.RowsCount

	outImageWidth := (imageWidth-filterSize+2*padding)/stride + 1
	outImageHeight := (imageHeight-filterSize+2*padding)/stride + 1

	outputSquare := outImageWidth * outImageHeight

	data := []float64{}

	iSquare := imageWidth * imageHeight
	fSquare := filterSize * filterSize
	iCube := iSquare * channels
	fCube := fSquare * channels

	for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
		for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {

			image := m.Data[imageIndex*iCube : (imageIndex+1)*iCube]
			filter := filters.Data[filterIndex*fCube : (filterIndex+1)*fCube]

			_, _, featureMap := Conv2D(
				imageWidth, imageHeight, image, // image
				filterSize, filterSize, filter, // filter
				channels,
				padding,
				stride,
			)

			AddScalarTo(featureMap, biases.Data[filterIndex])
			data = append(data, featureMap...)
		}
	}

	return NewMatrixResult(
		outputSquare,
		filtersCount,
		imagesCount,
		data,
		NewSource(func() {
			backpropConv(
				outImageWidth, outImageHeight, outMatrix.Grad,
				imageWidth, imageHeight, m.Data, m.Grad,
				filterSize, filters.Data, filters.Grad, biases.Grad,
				channels,
				imagesCount,
				filtersCount,
				padding,
			)
		}, m, filters, biases),
	)
}

func backpropConv(
	oW, oH int, deltas []float64,
	iW, iH int, inputs, iGrads []float64,
	filterSize int, filter, wGrads, bGrads []float64,
	channels,
	imagesCount,
	filtersCount,
	padding int,
) {
	_, _, filter = Rotate180(filterSize, filterSize, filtersCount*channels, filter)

	deltasPad, oWPadd, oHPadd := AddPadding(deltas, oW, oH, filtersCount, 2-padding)

	offsetPad := 0
	offset := 0

	oSquarePad := oWPadd * oHPadd
	iSquare := iW * iH
	oSquare := oW * oH

	fCube := filterSize * filterSize * channels
	iCube := iW * iH * channels

	//oCube := oSquarePad * channels
	for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
		for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {
			deltasPad := deltasPad[offsetPad : offsetPad+oSquarePad]
			offsetPad += oSquarePad

			deltas := deltas[offset : offset+oSquare]
			offset += oSquare

			bGrads[filterIndex] += Sum(deltas)

			filter := filter[filterIndex*fCube : (filterIndex+1)*fCube]
			wGrads := wGrads[filterIndex*fCube : (filterIndex+1)*fCube]
			iGrads := iGrads[imageIndex*iCube : (imageIndex+1)*iCube]
			inputs := inputs[imageIndex*iCube : (imageIndex+1)*iCube]

			wCoord := 0
			for izo := 0; izo < channels; izo++ {
				for iyo := 0; iyo < filterSize; iyo++ {
					for ixo := 0; ixo < filterSize; ixo++ {

						weight := filter[wCoord]
						wgradv := wGrads[wCoord]

						for row := 0; row < iH-filterSize; row++ {
							deltasPad := deltasPad[(iyo+row)*oWPadd+ixo : (iyo+row)*oWPadd+ixo+iW]
							iGrads := iGrads[izo*iSquare+(iyo+row)*iW : izo*iSquare+(iyo+row)*iW+iW]

							for dc, delta := range deltasPad {
								iGrads[dc] += delta * weight
							}
						}

						for row := 0; row < oH-filterSize; row++ {
							inputs := inputs[izo*iSquare+(iyo+row)*iW+ixo : izo*iSquare+(iyo+row)*iW+ixo+oW]
							deltas := deltas[(iyo+row)*oW+ixo : (iyo+row)*oW+ixo+oW]

							for dc, delta := range deltas {
								wgradv += inputs[dc] * delta
							}
						}

						wGrads[wCoord] += wgradv
						wCoord++
					}
				}
			}
		}
	}
}

const ReplaceMe = 1

func (m *Matrix) MatrixMultiply(b *Matrix) (outMatrix *Matrix) {
	rColsCount, rRowsCount, rData := MatrixMultiply(
		m.ColsCount, m.RowsCount, m.Data,
		b.ColsCount, b.RowsCount, b.Data,
	)

	return NewMatrixResult(rColsCount, rRowsCount, ReplaceMe, rData, NewSource(func() {
		oG := NewMatrix(
			outMatrix.ColsCount,
			outMatrix.RowsCount,
			outMatrix.ChanCount,
			outMatrix.Grad,
		)

		AddTo(m.Grad, oG.MatrixMultiply(b.Transpose()).Data)
		AddTo(b.Grad, m.Transpose().MatrixMultiply(oG).Data)
	}, m, b))
}

func (m *Matrix) AddRowVector(b *Matrix) (outMatrix *Matrix) {
	if b.RowsCount > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.RowsCount))
	}

	out := MatrixAddRowVector(m.ColsCount, m.RowsCount, m.Data, b.Data)

	return NewMatrixResult(m.ColsCount, m.RowsCount, ReplaceMe, out, NewSource(func() {
		// we have outMatrix.Grad
		// we need to calculate m.Grad and b.Grad
		// m.Grad = outMatrix.Grad, because outMatrix.Data = m.Data + Vector
		// b.Grad = [grads.Col(0).Dot(), grads.Col(1).Dot(), ... , grads.Col(n).Dot()]

		AddTo(m.Grad, outMatrix.Grad)

		for rowIndex := 0; rowIndex < m.RowsCount; rowIndex++ {
			AddTo(b.Grad, MatrixRowFloats(m.ColsCount, rowIndex, outMatrix.Grad))
		}
	}, m, b))
}

func (m *Matrix) Tanh() (outMatrix *Matrix) {
	return NewMatrixResult(m.ColsCount, m.RowsCount, m.ChanCount, Tanh(m.Data), NewSource(func() {
		for i, v := range outMatrix.Data {
			m.Grad[i] += outMatrix.Grad[i] * (1 - v*v)
		}
	}, m))
}

func (m *Matrix) Sigmoid() (outMatrix *Matrix) {
	return NewMatrixResult(m.ColsCount, m.RowsCount, m.ChanCount, Sigmoid(m.Data), NewSource(func() {
		for i, v := range outMatrix.Data {
			m.Grad[i] += outMatrix.Grad[i] * v * (1 - v)
		}
	}, m))
}

func (m *Matrix) Relu() (outMatrix *Matrix) {
	return NewMatrixResult(m.ColsCount, m.RowsCount, m.ChanCount, Relu(m.Data), NewSource(func() {
		for i, v := range outMatrix.Data {
			if v > 0 {
				m.Grad[i] += outMatrix.Grad[i]
			}
		}
	}, m))
}

func (m *Matrix) IsDimensionsEqual(t *Matrix) bool {
	return true &&
		m.ColsCount == t.ColsCount &&
		m.RowsCount == t.RowsCount &&
		m.ChanCount == t.ChanCount
}

func (m *Matrix) Regression(targets *Matrix) (outMatrix *Matrix) {
	if !m.IsDimensionsEqual(targets) {
		panic(fmt.Sprintf(
			"invalid targets dimensions: expected %v, actual %v",
			[3]int{m.ColsCount, m.RowsCount, m.ChanCount},
			[3]int{targets.ColsCount, targets.RowsCount, targets.ChanCount},
		))
	}

	r := 0.0
	for i, t := range targets.Data {
		r += math.Pow(m.Data[i]-t, 2)
	}
	r *= 0.5

	return NewMatrixResult(1, 1, ReplaceMe, []float64{r}, NewSource(func() {
		for i, t := range targets.Data {
			m.Grad[i] += m.Data[i] - t
		}
	}, m))
}

const minimalNonZeroFloat = 0.000000000000000000001

func (m *Matrix) Classification(targets *Matrix) (outMatrix *Matrix) {
	if !m.IsDimensionsEqual(targets) {
		panic(fmt.Sprintf(
			"invalid targets dimensions: expected %v, actual %v",
			[3]int{m.ColsCount, m.RowsCount, m.ChanCount},
			[3]int{targets.ColsCount, targets.RowsCount, targets.ChanCount},
		))
	}

	byClassesVal := make([]float64, m.RowsCount)
	for rowIndex := 0; rowIndex < m.RowsCount; rowIndex++ {
		row := MatrixRowFloats(m.ColsCount, rowIndex, targets.Data)

		for i, t := range row {
			if t == 1 {
				if m.Data[rowIndex*m.ColsCount+i] <= 0 {
					byClassesVal[rowIndex] = -math.Log(minimalNonZeroFloat)
				} else {
					byClassesVal[rowIndex] = -math.Log(m.Data[rowIndex*m.ColsCount+i])
				}
				break
			}
		}
	}

	// classification loss by each input in batch
	return NewMatrixResult(1, m.RowsCount, ReplaceMe, byClassesVal, NewSource(func() {
		for i := 0; i < len(targets.Data); i++ {
			m.Grad[i] += outMatrix.Grad[0] * (m.Data[i] - targets.Data[i])
		}

		//for row := 0; row < m.RowsCount; row++ {
		//	g := outMatrix.Grad[row]
		//	for col := 0; col < m.ColsCount; col++ {
		//		i := row*m.ColsCount + col
		//
		//		o := m.Data[i]
		//		t := targets.Data[i]
		//
		//		m.Grad[i] += g * (-(t / o) + ((1 - t) / (1 - o)))
		//	}
		//}
	}, m))
}

func (m *Matrix) Sum() (outMatrix *Matrix) {
	return NewMatrixResult(1, 1, 1, []float64{Sum(m.Data)}, NewSource(func() {
		AddScalarTo(m.Grad, outMatrix.Grad[0])
	}, m))
}

func (m *Matrix) Mean() (outMatrix *Matrix) {
	sum, k := Sum(m.Data), 1/float64(len(m.Data))
	return NewMatrixResult(1, 1, 1, []float64{sum * k}, NewSource(func() {
		AddScalarTo(m.Grad, outMatrix.Grad[0]*k)
	}, m))
}

func (m *Matrix) Exp() (outMatrix *Matrix) {
	return NewMatrixResult(m.ColsCount, m.RowsCount, m.ChanCount, Exp(m.Data), NewSource(func() {
		for i := range m.Grad {
			m.Grad[i] += outMatrix.Grad[i] * outMatrix.Data[i]
		}
	}, m))
}

func (m *Matrix) Log() (outMatrix *Matrix) {
	return NewMatrixResult(m.ColsCount, m.RowsCount, m.ChanCount, Log(m.Data), NewSource(func() {
		for i := range m.Grad {
			//m.Grad[i] += outMatrix.Grad[i] * 1 / m.Data[i]
			m.Grad[i] += outMatrix.Grad[i] / m.Data[i]
		}
	}, m))
}

func (m *Matrix) Mul(f float64) (outMatrix *Matrix) {
	return NewMatrixResult(m.ColsCount, m.RowsCount, m.ChanCount, Mul(m.Data, f), NewSource(func() {
		for i := range m.Grad {
			m.Grad[i] += outMatrix.Grad[i] * f
		}
	}, m))
}

func (m *Matrix) SoftmaxRows() (outMatrix *Matrix) {
	exps := Exp(m.Data)

	out := make([]float64, len(m.Data))
	copy(out, exps)

	sumByRows := make([]float64, m.RowsCount)
	for row := 0; row < m.RowsCount; row++ {
		offset := row * m.ColsCount
		sumByRows[row] = Sum(out[offset : offset+m.ColsCount])
		for col := 0; col < m.ColsCount; col++ {
			out[offset+col] /= sumByRows[row]
		}
	}

	outMatrix = NewMatrixResult(m.ColsCount, m.RowsCount, ReplaceMe, out, NewSource(func() {
		for i, g := range outMatrix.Grad {
			m.Grad[i] += exps[i] * g
		}
	}, m))

	return
}

func (m *Matrix) resetGrad() {
	Fill(m.Grad, 0)

	m.backwardCalled = false
	if m.From != nil {
		for _, prev := range m.From.Parents {
			prev.resetGrad()
		}
	}
}

func (m *Matrix) ResetGrad() {
	m.resetGrad()
}

func (m *Matrix) backward() {
	if m.backwardCalled {
		return
	}

	m.backwardCalled = true

	if m.From != nil {
		if m.From.Callback != nil {
			m.From.Callback()
		}

		for _, prev := range m.From.Parents {
			prev.backward()
		}
	}
}

func (m *Matrix) Backward() {
	Fill(m.Grad, 1)
	m.backward()
}

func (m *Matrix) GetDims() []int {
	return []int{m.ColsCount, m.RowsCount, m.ChanCount}
}
