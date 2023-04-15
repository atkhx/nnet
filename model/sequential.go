package model

func NewSequential(iSize int, layers []Layer) *Sequential {
	return &Sequential{
		iSize:  iSize,
		layers: layers,
	}
}

type Sequential struct {
	iSize int

	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers []Layer
	lossFn LossFn
}

func (s *Sequential) Compile() {
	s.inputs = make([]float64, s.iSize)
	s.iGrads = make([]float64, s.iSize)

	inputs, iGrads := s.inputs, s.iGrads

	for _, layer := range s.layers {
		inputs, iGrads = layer.Compile(inputs, iGrads)
	}

	s.output = inputs
	s.oGrads = iGrads
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
