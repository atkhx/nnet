package main

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/atkhx/nnet/data"
)

func s(f float64) *data.Scalar {
	return data.NewScalar(f)
}

func sRand() *data.Scalar {
	return data.NewScalar(rand.Float64())
}

var (
	inputSize  = 2
	batchSize  = 4
	layer1Size = 2
	layer2Size = 1
)

func main() {
	rand.Seed(1)
	inputs := []*data.Scalar{
		// 2 - input size
		// 4 - batch size
		s(0), s(0),
		s(1), s(0),
		s(0), s(1),
		s(1), s(1),
	}

	targets := []*data.Scalar{
		s(0),
		s(1),
		s(1),
		s(0),
	}

	weights1 := []*data.Scalar{}
	for i := 0; i < inputSize; i++ { // rows count
		for j := 0; j < layer1Size; j++ { // cols count
			weights1 = append(weights1, sRand())
		}
	}

	b1 := s(0.0)

	weights2 := []*data.Scalar{}
	for i := 0; i < layer1Size; i++ { // rows count
		for j := 0; j < layer2Size; j++ { // cols count
			weights2 = append(weights2, sRand())
		}
	}

	b2 := s(0.0)

	updateWeights := [][]*data.Scalar{weights1, weights2, {b1, b2}}

	lastIndex := 999
	for i := 0; i < 1000; i++ {
		outputs1 := matrixMultiply(
			inputSize, batchSize,
			layer1Size, inputSize,
			inputs,
			weights1,
		)

		//for j, v := range outputs1 {
		//	outputs1[j] = v.Add(b1)
		//}

		output1Activated := tanh(outputs1)

		outputs2 := matrixMultiply(
			layer1Size, batchSize,
			layer2Size, layer1Size,
			output1Activated,
			weights2,
		)

		//for j, v := range outputs2 {
		//	outputs2[j] = v.Add(b2)
		//}

		output2Activated := tanh(outputs2)

		loss := regression(output2Activated, targets)
		if i == 0 || i%100 == 0 {
			fmt.Println("loss", loss.Data)
		}

		if i == lastIndex {
			fmt.Println(strings.Repeat("-", 40))
			for _, v := range output2Activated {
				fmt.Println(v.Data)
			}
			fmt.Println(strings.Repeat("-", 40))
		}

		loss.ResetGrad()
		loss.Backward()

		for _, weights := range updateWeights {
			for _, p := range weights {
				p.Data -= 0.1 * p.Grad
			}
		}
	}
}

func tanh(a []*data.Scalar) []*data.Scalar {
	b := []*data.Scalar{}
	for _, v := range a {
		b = append(b, v.Tanh())
	}
	return b
}

func matrixMultiply(
	aColsCount, aRowsCount,
	bColsCount, bRowsCount int,
	a, b []*data.Scalar,
) []*data.Scalar {
	out := make([]*data.Scalar, bColsCount*aRowsCount)

	getColumnFromB := func(colIndex int) []*data.Scalar {
		res := make([]*data.Scalar, bRowsCount)
		for rowIndex := 0; rowIndex < bRowsCount; rowIndex++ {
			res[rowIndex] = b[rowIndex*bColsCount+colIndex]
		}

		return res
	}

	for weightIndex := 0; weightIndex < bColsCount; weightIndex++ {
		bValues := getColumnFromB(weightIndex)

		for inputIndex := 0; inputIndex < aRowsCount; inputIndex++ {
			aOffset := inputIndex * aColsCount
			aValues := a[aOffset : aOffset+aColsCount]

			abDot := s(0.0)
			for i, bV := range bValues {
				abDot = abDot.Add(bV.Mul(aValues[i]))
			}

			out[inputIndex*bColsCount+weightIndex] = abDot
		}
	}

	return out
}

func regression(d, targets []*data.Scalar) *data.Scalar {
	r := s(0.0)
	for i, t := range targets {
		r = r.Add(d[i].Sub(t).Pow(s(2)))
	}
	return r.Mul(s(0.5))
}
