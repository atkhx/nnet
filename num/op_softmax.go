package num

import "sync"

func (input *Data) Softmax() *Data {
	chunkSize := input.Dims.W
	chunksCount := len(input.Data) / chunkSize

	output := input.Copy()
	output.SetOperation("softmax")

	wg := sync.WaitGroup{}

	output.calcData = func() {
		output.Data.CopyFrom(input.Data)

		wg.Add(chunksCount)
		for i := 0; i < len(output.Data); i += chunkSize {
			go func(i int) {
				output.Data[i : i+chunkSize].Softmax()
				wg.Done()
			}(i)
		}
		wg.Wait()
	}

	output.calcGrad = func() {
		wg.Add(chunksCount)

		for b := 0; b < len(output.Data); b += chunkSize {
			go func(b int) {
				oGrad := output.Grad[b : b+chunkSize]
				iGrad := input.Grad[b : b+chunkSize]

				softmax := output.Data[b : b+chunkSize]

				for i, softmaxI := range softmax {
					gI := oGrad[i] * softmaxI
					for j, softmaxJ := range softmax {
						if i == j {
							iGrad[j] += gI * (1 - softmaxI)
						} else {
							iGrad[j] -= gI * softmaxJ
						}
					}
				}

				wg.Done()
			}(b)
		}

		wg.Wait()
	}

	return output
}
