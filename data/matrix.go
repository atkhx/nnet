package data

import (
	"fmt"
	"math"
	"math/rand"
)

func NewMatrixRandom(cols, rows int) *Matrix {
	data := make([]float64, cols*rows)
	for i := range data {
		data[i] = rand.Float64()
	}
	return NewMatrix(cols, rows, data)
}

func NewMatrix(cols, rows int, data []float64) *Matrix {
	if len(data) != cols*rows {
		panic(fmt.Sprintf("invalid dimensions: data length %d, cols*rows %d", len(data), cols*rows))
	}

	return &Matrix{
		Dims: [2]int{cols, rows},
		Data: data,
	}
}

func NewMatrixResult(cols, rows int, data []float64, from *Source) (outMatrix *Matrix) {
	outMatrix = NewMatrix(cols, rows, data)
	outMatrix.From = from
	return
}

type Matrix struct {
	Dims [2]int
	Data []float64
	Grad []float64
	From *Source

	backwardCalled bool
}

func (m *Matrix) GradsMatrix() *Matrix {
	return NewMatrix(m.Dims[0], m.Dims[1], m.Grad)
}

func (m *Matrix) gradRowFloats(rowIndex int) []float64 {
	colsCount := m.Dims[0]
	rowOffset := rowIndex * colsCount

	return m.Grad[rowOffset : rowOffset+colsCount]
}

func (m *Matrix) Transpose() (outMatrix *Matrix) {
	outMatrix = NewMatrix(MatrixTranspose(m.Dims[0], m.Dims[1], m.Data))
	outMatrix.From = NewSource(func() {
		m.InitGrad()
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
		m.InitGrad()
		b.InitGrad()

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

		m.InitGrad()
		b.InitGrad()

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
		m.InitGrad()

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
		m.InitGrad()

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
		m.InitGrad()

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
		m.InitGrad()

		for i, t := range targets.Data {
			m.Grad[i] += m.Data[i] - t
		}
	}, m))
}

const minimalNonZeroFloat = 0.000000000000000000001

func (m *Matrix) SumByRows() (outMatrix *Matrix) {
	out := make([]float64, m.Dims[1])
	for row := 0; row < m.Dims[1]; row++ {
		for col := 0; col < m.Dims[0]; col++ {
			out[row] += m.Data[row*m.Dims[0]+col]
		}
	}

	return NewMatrix(1, m.Dims[1], out)
}

func (m *Matrix) Classification(targets *Matrix) (outMatrix *Matrix) {
	if m.Dims != targets.Dims {
		panic(fmt.Sprintf("invalid targets dimensions: expected %v, actual %v", m.Dims, targets.Dims))
	}

	byClassesVal := make([]float64, m.Dims[1])
	for rowIndex := 0; rowIndex < m.Dims[1]; rowIndex++ {
		row := MatrixRowFloats(m.Dims[0], rowIndex, targets.Data)

		for i, t := range row {
			if t == 1 {
				byClassesVal[rowIndex] = -math.Log(m.Data[rowIndex*m.Dims[0]+i])
				break
			}
		}
	}

	// classification loss by each input in batch
	return NewMatrixResult(1, m.Dims[1], byClassesVal, NewSource(func() {
		m.InitGrad()

		for row := 0; row < m.Dims[1]; row++ {
			g := outMatrix.Grad[row]
			for col := 0; col < m.Dims[0]; col++ {
				i := row*m.Dims[0] + col

				o := m.Data[i]
				t := targets.Data[i]

				m.Grad[i] += g * (-(t / o) + ((1 - t) / (1 - o)))
			}
		}
	}, m))
}

func (m *Matrix) Sum() (outMatrix *Matrix) {
	out := 0.0
	for _, v := range m.Data {
		out += v
	}

	return NewMatrixResult(1, 1, []float64{out}, NewSource(func() {
		m.InitGrad()
		for i := range m.Grad {
			m.Grad[i] += outMatrix.Grad[0]
		}
	}, m))
}

func (m *Matrix) Mean() (outMatrix *Matrix) {
	out := 0.0
	for _, v := range m.Data {
		out += v
	}

	k := 1 / float64(len(m.Data))
	return NewMatrixResult(1, 1, []float64{out * k}, NewSource(func() {
		m.InitGrad()
		for i := range m.Grad {
			m.Grad[i] += outMatrix.Grad[0] * k
		}
	}, m))
}

func (m *Matrix) Exp() (outMatrix *Matrix) {
	max := 0.0
	for i, v := range m.Data {
		if i == 0 || max < v {
			max = v
		}
	}

	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = math.Exp(v - max)
	}

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		m.InitGrad()
		for i := range m.Grad {
			m.Grad[i] += outMatrix.Grad[i] * out[i]
		}
	}, m))
}

func (m *Matrix) Log() (outMatrix *Matrix) {
	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = math.Log(v)
	}

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		m.InitGrad()
		for i := range m.Grad {
			//m.Grad[i] += outMatrix.Grad[i] * 1 / m.Data[i]
			m.Grad[i] += outMatrix.Grad[i] / m.Data[i]
		}
	}, m))
}

func (m *Matrix) Mul(f float64) (outMatrix *Matrix) {
	out := make([]float64, len(m.Data))
	copy(out, m.Data)

	for i := range out {
		out[i] *= f
	}

	return NewMatrixResult(m.Dims[0], m.Dims[1], out, NewSource(func() {
		m.InitGrad()
		for i := range m.Grad {
			m.Grad[i] += -outMatrix.Grad[i] * f
		}
	}, m))
}

func (m *Matrix) SoftmaxRows() (outMatrix *Matrix) {

	ex := m.Exp()

	out := make([]float64, len(ex.Data))
	copy(out, ex.Data)

	sumByRows := make([]float64, m.Dims[1])
	for row := 0; row < ex.Dims[1]; row++ {
		offset := row * ex.Dims[0]

		//sum, max := 0.0, 0.0
		//for col := 0; col < m.Dims[0]; col++ {
		//	if col == 0 || m.Data[offset+col] > max {
		//		max = m.Data[offset+col]
		//	}
		//}
		//for col := 0; col < m.Dims[0]; col++ {
		//	out[offset+col] = math.Exp(m.Data[offset+col] - max)
		//	sum += out[offset+col]
		//}

		for col := 0; col < ex.Dims[0]; col++ {
			sumByRows[row] += out[offset+col]
		}

		for col := 0; col < ex.Dims[0]; col++ {
			out[offset+col] /= sumByRows[row]
		}
	}

	outMatrix = NewMatrixResult(ex.Dims[0], ex.Dims[1], out, NewSource(func() {
		ex.InitGrad()
		for row := 0; row < ex.Dims[1]; row++ {
			for col := 0; col < ex.Dims[0]; col++ {
				i := row*ex.Dims[0] + col

				ex.Grad[i] += outMatrix.Grad[i] / sumByRows[row]
			}
		}

		//for i, o := range out {
		//	ex.Grad[i] += outMatrix.Grad[i] * (o * (1 - o))
		//}
	}, ex))

	return
}

func (m *Matrix) InitGrad() {
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
	m.InitGrad()
	for i := range m.Grad {
		m.Grad[i] = 1
	}
	m.backward()
}

func NewOneHotVectorsMatrix(colsCount int, hots ...int) (outMatrix *Matrix) {
	rowsCount := len(hots)

	data := make([]float64, 0, colsCount*len(hots))

	for row := 0; row < rowsCount; row++ {
		vector := make([]float64, colsCount)
		vector[hots[row]] = 1

		data = append(data, vector...)
	}

	return NewMatrix(colsCount, rowsCount, data)
}
