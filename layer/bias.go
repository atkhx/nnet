package layer

func NewBias(inputs, iGrads []float64) *Bias {
	return &Bias{
		weights: make([]float64, len(inputs)),
		wGrads:  make([]float64, len(inputs)),

		inputs: inputs,
		iGrads: iGrads,

		output: make([]float64, len(inputs)),
		oGrads: make([]float64, len(inputs)),
	}
}

type Bias struct {
	// internal buffers
	weights []float64
	wGrads  []float64

	// buffers from the previous layer
	inputs []float64
	iGrads []float64

	// buffers to the next layer
	output []float64
	oGrads []float64
}

func (l *Bias) Buffers() (output, oGrads []float64) {
	return l.output, l.oGrads
}

func (l *Bias) Forward() {
	for i, w := range l.weights {
		l.output[i] = w + l.inputs[i]
	}
}

func (l *Bias) Backward() {
	for i, g := range l.oGrads {
		l.iGrads[i] += g
		l.wGrads[i] += g
	}
}

func (l *Bias) ResetGrads() {
	for i := range l.wGrads {
		l.wGrads[i] = 0
		l.oGrads[i] = 0
	}
}

func (l *Bias) ForUpdate() [][2][]float64 {
	return [][2][]float64{
		{l.weights, l.wGrads},
	}
}
