package data

import (
	"fmt"
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

func (m *Data) InitVectorWithData(w int, data []float64) {
	m.Dims = []int{w, 1, 1}
	m.Data = data
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

func (m *Data) PrintlnCube() {
	var w, h, d int
	m.ExtractDimensions(&w, &h, &d)

	fmt.Printf("data dims: %d x %d x %d \n", w, h, d)
	for z := 0; z < d; z++ {
		fmt.Println("z:", z)

		for y := 0; y < h; y++ {
			fmt.Print("      ")
			for x := 0; x < w; x++ {
				fmt.Print(m.Data[z*w*h+y*w+x], " ")
			}
			fmt.Println()
		}
	}
}

func (m *Data) Rotate180() {
	var w, h, d int
	m.ExtractDimensions(&w, &h, &d)

	f := make([]float64, len(m.Data))

	for z := 0; z < d; z++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f[z*w*h+(h-y-1)*w+(w-x-1)] = m.Data[z*w*h+y*w+x]
			}
		}
	}

	m.Data = f
}

func (m *Data) RotateRight90() {
	var w, h, d int
	m.ExtractDimensions(&w, &h, &d)

	f := make([]float64, len(m.Data))

	for z := 0; z < d; z++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f[z*w*h+x*h+(h-y-1)] = m.Data[z*w*h+y*w+x]
			}
		}
	}

	m.Dims[0] = h
	m.Dims[1] = w

	m.Data = f
}

func (m *Data) RotateLeft90() {
	var w, h, d int
	m.ExtractDimensions(&w, &h, &d)

	f := make([]float64, len(m.Data))

	for z := 0; z < d; z++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f[z*w*h+(w-x-1)*h+y] = m.Data[z*w*h+y*w+x]
			}
		}
	}

	m.Dims[0] = h
	m.Dims[1] = w

	m.Data = f
}

func (m *Data) AddPadding(padding int) *Data {
	var ow, oh, od int
	var pw, ph, pd int
	m.ExtractDimensions(&ow, &oh, &od)

	pd = od
	pw = ow + 2*padding
	ph = oh + 2*padding

	res := make([]float64, pw*ph*pd)

	for z := 0; z < pd; z++ {
		for y := padding; y < ph-padding; y++ {
			copy(
				res[z*ph*pw+y*pw+padding:z*ph*pw+y*pw+padding+ow],
				m.Data[z*oh*ow+(y-padding)*ow:z*oh*ow+(y-padding)*ow+ow],
			)
		}
	}

	vec := &Data{}
	vec.InitCubeWithData(pw, ph, pd, res)

	return vec
}

func (m *Data) RemovePadding(padding int) *Data {
	var ow, oh, od int
	var pw, ph, pd int
	m.ExtractDimensions(&ow, &oh, &od)

	pd = od
	pw = ow - 2*padding
	ph = oh - 2*padding

	res := make([]float64, pw*ph*pd)

	for z := 0; z < pd; z++ {
		for y := 0; y < ph; y++ {
			copy(
				res[z*ph*pw+y*pw:z*ph*pw+y*pw+pw],
				m.Data[z*oh*ow+(y+padding)*ow+padding:z*oh*ow+(y+padding)*ow+padding+pw],
			)
		}
	}

	vec := &Data{}
	vec.InitCubeWithData(pw, ph, pd, res)

	return vec
}
