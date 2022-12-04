package data

type Data struct {
	Dims []int
	Data []float64
}

func (m *Data) GetDimensionsCount() int {
	return len(m.Dims)
}

func (m *Data) ExtractDimensions(dims ...*int) {
	dimensionsCount := m.GetDimensionsCount()
	if dimensionsCount == 0 {
		return
	}

	for i, dim := range dims {
		if i < dimensionsCount {
			*dim = m.Dims[i]
		} else {
			*dim = 1
		}
	}
}

// Fill data methods

func (m *Data) FillZero() {
	Fill(m.Data, 0)
}

func (m *Data) Fill(v float64) {
	Fill(m.Data, v)
}

func (m *Data) FillRandom(min, max float64) {
	FillRandom(m.Data, min, max)
}

// initialize methods

func (m *Data) Copy() *Data {
	r := m.CopyZero()

	copy(r.Data, m.Data)
	return r
}

func (m *Data) CopyZero() *Data {
	r := &Data{
		Dims: make([]int, len(m.Dims)),
		Data: make([]float64, len(m.Data)),
	}

	copy(r.Dims, m.Dims)
	return r
}

// math methods

func (m *Data) Dot(floats []float64) float64 {
	return Dot(m.Data, floats)
}

func (m *Data) Add(src ...[]float64) {
	AddTo(m.Data, src...)
}

func (m *Data) GetMaxValue() float64 {
	return GetMaxValue(m.Data)
}

func (m *Data) GetMinValue() float64 {
	return GetMinValue(m.Data)
}

func (m *Data) GetMinMaxValues() (float64, float64) {
	return GetMinMaxValues(m.Data)
}

func (m *Data) GetMinMaxValuesInRange(from, to int) (float64, float64) {
	return GetMinMaxValuesInRange(m.Data, from, to)
}

func (m *Data) SumElements() float64 {
	return SumElements(m.Data)
}

// special methods

func (m *Data) Rotate180() *Data {
	res := m.Copy()

	var w, h, d int
	res.ExtractDimensions(&w, &h, &d)

	for z := 0; z < d; z++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				res.Data[z*w*h+(h-y-1)*w+(w-x-1)] = m.Data[z*w*h+y*w+x]
			}
		}
	}

	return res
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
	if padding == 0 {
		return m
	}

	var ow, oh, od int
	var pw, ph, pd int
	m.ExtractDimensions(&ow, &oh, &od)

	pd = od
	pw = ow + 2*padding
	ph = oh + 2*padding

	res := make([]float64, pw*ph*pd)

	phpw := ph * pw
	ohow := oh * ow

	for z := 0; z < pd; z++ {
		for y := padding; y < ph-padding; y++ {
			copy(
				res[z*phpw+y*pw+padding:z*phpw+y*pw+padding+ow],
				m.Data[z*ohow+(y-padding)*ow:z*ohow+(y-padding)*ow+ow],
			)
			//copy(
			//	res[z*ph*pw+y*pw+padding:z*ph*pw+y*pw+padding+ow],
			//	m.Data[z*oh*ow+(y-padding)*ow:z*oh*ow+(y-padding)*ow+ow],
			//)
		}
	}

	vec := &Data{}
	vec.Init3DWithData(pw, ph, pd, res)

	return vec
}

func (m *Data) RemovePadding(padding int) *Data {
	if padding == 0 {
		return m
	}

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
	vec.Init3DWithData(pw, ph, pd, res)

	return vec
}
