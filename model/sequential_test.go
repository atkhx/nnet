package model

import (
	"fmt"
	"testing"

	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func TestSequential(t *testing.T) {
	allInputs := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	allTargets := [][]float64{{0}, {1}, {1}, {0}}
	learningRate := 0.01

	model := NewSequential(2)

	// FC Block 1
	model.RegisterLayer(func(inputs, iGrads []float64) Layer {
		return layer.NewFC(10, inputs, iGrads)
	})

	model.RegisterLayer(func(inputs, iGrads []float64) Layer {
		return layer.NewBias(inputs, iGrads)
	})

	model.RegisterLayer(func(inputs, iGrads []float64) Layer {
		return layer.NewSigmoid(inputs, iGrads)
	})

	// FC Block 2
	model.RegisterLayer(func(inputs, iGrads []float64) Layer {
		return layer.NewFC(1, inputs, iGrads)
	})

	model.RegisterLayer(func(inputs, iGrads []float64) Layer {
		return layer.NewBias(inputs, iGrads)
	})

	model.RegisterLayer(func(inputs, iGrads []float64) Layer {
		return layer.NewSigmoid(inputs, iGrads)
	})

	var loss float64
	for i := 0; i < 100_000; i++ {
		for j := range allInputs {
			model.Forward(allInputs[j])

			loss += num.Regression(model.GetOutput(), allTargets[j], model.GetOGrads())

			model.Backward()
			model.Update(learningRate)
		}

		if i > 0 && i%1000 == 0 {
			loss /= float64(len(allInputs) * 1000)
			fmt.Println("loss", loss)
			loss = 0
		}
	}
}
