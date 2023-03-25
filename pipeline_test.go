package nnet

import (
	"fmt"
	"testing"
)

func TestPipeline_Forward(t *testing.T) {
	pipeline := Pipeline{}
	pipeline.layers = append(pipeline.layers, NewFullyConnectedLayer(2, 4, 1))

	inputs := []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
	}

	targets := []float64{
		0,
		1,
		1,
		0,
	}

	pipeline.data = append(pipeline.data, make([]float64, len(inputs)))
	pipeline.data = append(pipeline.data, make([]float64, len(targets)))

	output := pipeline.Forward(inputs)
	fmt.Println(output)
}
