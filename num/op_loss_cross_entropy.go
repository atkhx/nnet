package num

import "math"

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
