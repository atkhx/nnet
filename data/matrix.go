package data

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
