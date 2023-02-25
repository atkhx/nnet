package data

import (
	"fmt"
	"math"
)

func NewMatrix(cols, rows int, data []float64) *Matrix {
	if len(data) != cols*rows {
		panic("invalid dimensions")
	}

	return &Matrix{
		Dims: [2]int{cols, rows},
		Data: data,
	}
}

func NewMatrixResult(cols, rows int, data []float64, from *Source) *Matrix {
	if len(data) != cols*rows {
		panic("invalid dimensions")
	}

	return &Matrix{
		Dims: [2]int{cols, rows},
		Data: data,
		From: from,
	}
}

type Matrix struct {
	Dims [2]int
	Data []float64
	Grad []float64
	From *Source

	backwardCalled bool
}

func (m *Matrix) gradRowFloats(rowIndex int) []float64 {
	colsCount := m.Dims[0]
	rowOffset := rowIndex * colsCount

	return m.Grad[rowOffset : rowOffset+colsCount]
}

func (m *Matrix) Transpose() (outMatrix *Matrix) {
	outMatrix = NewMatrix(MatrixTranspose(m.Dims[0], m.Dims[1], m.Data))
	outMatrix.From = NewSource(func() {
		m.initGrad()
		oGT := NewMatrix(outMatrix.Dims[0], outMatrix.Dims[1], outMatrix.Grad).Transpose()
		for i, v := range oGT.Data {
			m.Grad[i] += v
		}
	}, m)
	return
}

func (m *Matrix) MatrixMultiply(b *Matrix) (outMatrix *Matrix) {
	rColsCount, rRowsCount, rData := MatrixMultiply(m.Dims[0], m.Dims[1], m.Data, b.Dims[0], b.Dims[1], b.Data)

	return NewMatrixResult(rColsCount, rRowsCount, rData, NewSource(func() {
		m.initGrad()
		b.initGrad()

		oG := NewMatrix(outMatrix.Dims[0], outMatrix.Dims[1], outMatrix.Grad)
		bT := b.Transpose()
		mT := m.Transpose()

		mdg := oG.MatrixMultiply(bT)
		for i, v := range mdg.Data {
			m.Grad[i] += v
		}

		bgd := mT.MatrixMultiply(oG)
		for i, v := range bgd.Data {
			b.Grad[i] += v
		}
	}, m, b))
}

func (m *Matrix) AddRowVector(b *Matrix) (outMatrix *Matrix) {
	if b.Dims[1] > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.Dims[1]))
	}

	out := MatrixAddRowVector(m.Dims[0], m.Dims[1], m.Data, b.Data)

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		// we have outMatrix.Grad
		// we need to calculate m.Grad and b.Grad
		// m.Grad = outMatrix.Grad, because outMatrix.Data = m.Data + Vector
		// b.Grad = [grads.Col(0).Dot(), grads.Col(1).Dot(), ... , grads.Col(n).Dot()]

		m.initGrad()
		b.initGrad()

		// todo addMatrixFunc
		for i, v := range outMatrix.Grad {
			m.Grad[i] += v
		}

		// todo use grad.Transpose().DotRows()
		for rowIndex := 0; rowIndex < m.Dims[1]; rowIndex++ {
			gradsRow := MatrixRowFloats(m.Dims[0], rowIndex, outMatrix.Grad)
			for i, v := range gradsRow {
				b.Grad[i] += v
			}
		}
	}, m, b))
}

func (m *Matrix) Tanh() (outMatrix *Matrix) {
	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = math.Tanh(v)
	}

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		m.initGrad()

		for i, v := range out {
			m.Grad[i] += outMatrix.Grad[i] * (1 - v*v)
		}
	}, m))
}

func (m *Matrix) Sigmoid() (outMatrix *Matrix) {
	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = 1 / (1 + math.Exp(-v))
	}

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		m.initGrad()

		for i, v := range out {
			m.Grad[i] += outMatrix.Grad[i] * v * (1 - v)
		}
	}, m))
}

func (m *Matrix) Relu() (outMatrix *Matrix) {
	out := make([]float64, len(m.Data))
	copy(out, m.Data)

	for i, v := range out {
		if v < 0 {
			out[i] = 0
		}
	}

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		m.initGrad()

		for i, v := range out {
			if v > 0 {
				m.Grad[i] += outMatrix.Grad[i]
			}
		}
	}, m))
}

func (m *Matrix) Regression(targets *Matrix) (outMatrix *Matrix) {
	if m.Dims != targets.Dims {
		panic(fmt.Sprintf("invalid targets dimensions: expected %v, actual %v", m.Dims, targets.Dims))
	}

	r := 0.0
	for i, t := range targets.Data {
		r += math.Pow(m.Data[i]-t, 2)
	}
	r *= 0.5

	return NewMatrixResult(1, 1, []float64{r}, NewSource(func() {
		m.initGrad()

		for i, t := range targets.Data {
			m.Grad[i] += m.Data[i] - t
		}
	}, m))
}

func (m *Matrix) initGrad() {
	if m.Grad == nil {
		m.Grad = make([]float64, len(m.Data))
	}
}

func (m *Matrix) resetGrad() {
	for i := range m.Grad {
		m.Grad[i] = 0
	}

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
	for i := range m.Grad {
		m.Grad[i] = 1
	}
	m.backward()
}
