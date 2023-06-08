package num

import "sync"

func (aData *Data) Softmax() *Data {
	chunkSize := aData.Dims.W
	chunksCount := len(aData.Data) / chunkSize

	wg := sync.WaitGroup{}

	output := aData.Copy()
	output.Name = "softmax"
	output.calcData = func() {
		output.Data.CopyFrom(aData.Data)

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
				iGrad := aData.Grad[b : b+chunkSize]

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
