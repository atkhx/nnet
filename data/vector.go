package data

func NewVector(data ...float64) *Data {
	res := &Data{}
	res.InitVector(len(data))

	copy(res.Data, data)
	return res
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
