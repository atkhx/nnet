package layer

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	// buffers from the previous layer
	inputs []float64
	iGrads []float64

	// buffers to the next layer
	output []float64
	oGrads []float64
}

func (l *ReLu) Compile(inputs, iGrads []float64) ([]float64, []float64) {
	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make([]float64, len(inputs))
	l.oGrads = make([]float64, len(inputs))

	return l.output, l.oGrads
}

func (l *ReLu) Forward() {
	for i, v := range l.inputs {
		if v > 0 {
			l.output[i] = v
		} else {
			l.output[i] = 0
		}
	}
}

func (l *ReLu) Backward() {
	for i, v := range l.output {
		if v > 0 {
			l.iGrads[i] += l.oGrads[i]
		}
	}
}

func (l *ReLu) ResetGrads() {
	for i := range l.oGrads {
		l.oGrads[i] = 0
	}
}
