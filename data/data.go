package data

import (
	"math/rand"
)

type Data struct {
	Dims []int
	Data []float64
}

func (m *Data) ExtractDimensions(dims ...*int) {
	dimsCount := len(m.Dims)
	for i, dim := range dims {
		if i < dimsCount {
			*dim = m.Dims[i]
		} else {
			*dim = 1
		}
	}
}

func (m *Data) Reset() {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = 0
	}
}

func (m *Data) Fill(v float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = v
	}
}

func (m *Data) FillRandom(min, max float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = min + (max-min)*rand.Float64()
	}
}

func (m *Data) InitVector(w int) {
	m.Dims = []int{w, 1, 1}
	m.Data = make([]float64, w)
}

func (m *Data) InitVectorRandom(w int, min, max float64) {
	m.InitVector(w)
	m.FillRandom(min, max)
}

func (m *Data) InitMatrix(w, h int) {
	m.Dims = []int{w, h, 1}
	m.Data = make([]float64, w*h)
}

func (m *Data) InitMatrixWithData(w, h int, data []float64) {
	m.Dims = []int{w, h, 1}
	m.Data = data
}

func (m *Data) InitMatrixRandom(w, h int, min, max float64) {
	m.InitMatrix(w, h)
	m.FillRandom(min, max)
}

func (m *Data) InitCube(w, h, d int) {
	m.Dims = []int{w, h, d}
	m.Data = make([]float64, w*h*d)
}

func (m *Data) InitCubeWithData(w, h, d int, data []float64) {
	m.Dims = []int{w, h, d}
	m.Data = data
}

func (m *Data) InitCubeRandom(w, h, d int, min, max float64) {
	m.InitCube(w, h, d)
	m.FillRandom(min, max)
}

func (m *Data) InitHiperCube(w, h, d, t int) {
	m.Dims = []int{w, h, d, t}
	m.Data = make([]float64, w*h*d*t)
}

func (m *Data) InitHiperCubeRandom(w, h, d, t int, min, max float64) {
	m.InitHiperCube(w, h, d, t)
	m.FillRandom(min, max)
}

func (m *Data) CopyZero() (r *Data) {
	r = &Data{}
	r.Dims = make([]int, len(m.Dims))
	r.Data = make([]float64, len(m.Data))
	copy(r.Dims, m.Dims) // copy struct

	return
}

func (m Data) Copy() (r *Data) {
	r = m.CopyZero()
	copy(r.Data, m.Data)
	return
}

func (m *Data) GetMinMaxValues(fromIndex, toIndex int) (min, max float64) {
	min, max = m.Data[fromIndex], m.Data[fromIndex]
	for i := fromIndex + 1; i < toIndex; i++ {
		if min > m.Data[i] {
			min = m.Data[i]
		}
		if max < m.Data[i] {
			max = m.Data[i]
		}
	}
	return
}
