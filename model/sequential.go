package model

func NewSequential(iSize int) *Sequential {
	return &Sequential{
		inputs: make([]float64, iSize),
		iGrads: make([]float64, iSize),
		output: nil,
		oGrads: nil,
	}
}

type Sequential struct {
	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers []Layer
	lossFn LossFn
}

func (s *Sequential) RegisterLayer(newLayer func(inputs, iGrads []float64) Layer) {
	var layer Layer
	if len(s.layers) == 0 {
		layer = newLayer(s.inputs, s.iGrads)
	} else {
		layer = newLayer(s.output, s.oGrads)
	}

	s.layers = append(s.layers, layer)
	s.output, s.oGrads = layer.Buffers()
}

func (s *Sequential) Forward(inputs, output []float64) {
	copy(s.inputs, inputs)
	for _, layer := range s.layers {
		layer.Forward()
	}
	copy(output, s.output)
}

func (s *Sequential) Backward(target []float64) {
	for i, t := range target {
		s.oGrads[i] = s.output[i] - t
	}

	for i := len(s.layers); i > 0; i-- {
		s.layers[i-1].Backward()
	}
}

func (s *Sequential) Update(learningRate float64) {
	for _, layer := range s.layers {
		if l, ok := layer.(Updatable); ok {
			for _, pair := range l.ForUpdate() {
				for j := range pair[1] {
					pair[0][j] -= pair[1][j] * learningRate
				}
			}
		}
	}

	for _, layer := range s.layers {
		if l, ok := layer.(WithGrads); ok {
			l.ResetGrads()
		}
	}

	for i := range s.iGrads {
		s.iGrads[i] = 0
	}

	for i := range s.oGrads {
		s.oGrads[i] = 0
	}
}
