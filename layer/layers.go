package layer

type Layers []Layer

func (s Layers) Compile(inputs, iGrads []float64) ([]float64, []float64) {
	for _, layer := range s {
		inputs, iGrads = layer.Compile(inputs, iGrads)
	}

	return inputs, iGrads
}

func (s Layers) Forward() {
	for _, layer := range s {
		layer.Forward()
	}
}

func (s Layers) Backward() {
	for i := len(s); i > 0; i-- {
		s[i-1].Backward()
	}
}

func (s Layers) ResetGrads() {
	for _, layer := range s {
		if l, ok := layer.(WithGrads); ok {
			l.ResetGrads()
		}
	}
}

func (s Layers) ForUpdate() [][2][]float64 {
	result := make([][2][]float64, 0, len(s))

	for _, layer := range s {
		if l, ok := layer.(Updatable); ok {
			result = append(result, l.ForUpdate()...)
		}
	}
	return result
}
