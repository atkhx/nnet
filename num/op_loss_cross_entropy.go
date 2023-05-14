package num

import "math"

func (input *Data) CrossEntropyPos(targets *Data) *Data {
	if targets.Dims.W != 1 {
		panic("target width must be equal 1")
	}

	if targets.Dims.H != input.Dims.H {
		panic("target height must be equal input height")
	}

	if targets.Dims.D != input.Dims.D {
		panic("target depth must be equal input depth")
	}

	oDims := targets.Dims
	chunkSize := input.Dims.W

	// just buffer to avoid memory allocations
	softmax := input.Data.CopyZero()

	output := New(oDims, input)
	output.calcData = func() {
		softmax.CopyFrom(input.Data)
		for i := 0; i < len(softmax); i += chunkSize {
			softmax[i : i+chunkSize].Softmax()
		}

		for rowIdx, correctIdx := range targets.Data {
			for i := 0; i < chunkSize; i++ {
				if i == int(correctIdx) {
					output.Data[rowIdx] = -math.Log(softmax[rowIdx*chunkSize+i])
				}
			}
		}
	}

	output.calcGrad = func() {
		for rowIdx, correctIdx := range targets.Data {
			for i := 0; i < chunkSize; i++ {
				j := rowIdx*chunkSize + i
				t := 0.0
				if i == int(correctIdx) {
					t = 1.0
				}
				input.Grad[j] += output.Grad[0] * (softmax[j] - t)
			}
		}
	}

	return output
}

func (input *Data) CrossEntropy(targets *Data) *Data {
	input.Dims.MustBeEqual(targets.Dims)
	oDims := input.Dims
	oDims.W = 1
	chunkSize := input.Dims.W

	// just buffer to avoid memory allocations
	softmax := input.Data.CopyZero()
	logLikelihood := input.Data.CopyZero()

	output := New(oDims, input)
	output.calcData = func() {
		softmax.CopyFrom(input.Data)
		for i := 0; i < len(softmax); i += chunkSize {
			softmax[i : i+chunkSize].Softmax()
		}

		for i, t := range targets.Data {
			logLikelihood[i] = -t * math.Log(softmax[i])
		}

		for i := range output.Data {
			output.Data[i] = logLikelihood[i*chunkSize : (i+1)*chunkSize].Sum()
		}
	}
	output.calcGrad = func() {
		for i, t := range targets.Data {
			input.Grad[i] += softmax[i] - t
		}
	}

	return output
}
