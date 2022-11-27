package data

func (m *Data) Init3D(w, h, d int) {
	m.Dims = []int{w, h, d}
	m.Data = make([]float64, w*h*d)
}

func (m *Data) Init3DWithData(w, h, d int, data []float64) {
	m.Dims = []int{w, h, d}
	m.Data = data
}

func (m *Data) Init3DRandom(w, h, d int, min, max float64) {
	m.Init3D(w, h, d)
	m.FillRandom(min, max)
}

func (m *Data) Init4D(w, h, d, t int) {
	m.Dims = []int{w, h, d, t}
	m.Data = make([]float64, w*h*d*t)
}

func (m *Data) Init4DRandom(w, h, d, t int, min, max float64) {
	m.Init4D(w, h, d, t)
	m.FillRandom(min, max)
}
