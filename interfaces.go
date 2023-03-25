package nnet

type Layer interface {
	Forward(inputs, output []float64)
	Backward(gradient []float64)
}

func fill(dst []float64, v float64) {
	for i := range dst {
		dst[i] = v
	}
}
