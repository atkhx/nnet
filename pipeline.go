package nnet

type Pipeline struct {
	data [][]float64
	grad [][]float64

	layers []func(inputs, output []float64)
}

func (n *Pipeline) Forward(inputs []float64) []float64 {
	copy(n.data[0], inputs)

	for i := 1; i < len(n.data); i++ {
		fill(n.data[i], 0.0)
	}

	for i, layer := range n.layers {
		layer(n.data[i], n.data[i+1])
	}

	output := make([]float64, len(n.data[len(n.data)-1]))
	copy(output, n.data[len(n.data)-1])

	return output
}

func (n *Pipeline) Backward() {
	for i := len(n.grad) - 1; i > 0; i-- {
		fill(n.grad[i], 0.0)
	}
	fill(n.grad[len(n.grad)-1], 1.0)

}
