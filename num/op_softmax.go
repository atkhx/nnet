package num

func (input *Data) Softmax() *Data {
	chunkSize := input.Dims.W

	output := input.Copy()
	output.calcData = func() {
		output.Data.CopyFrom(input.Data)
		for i := 0; i < len(output.Data); i += chunkSize {
			output.Data[i : i+chunkSize].Softmax()
		}
	}

	output.calcGrad = func() {
		for b := 0; b < len(output.Data); b += chunkSize {
			oGrad := output.Grad[b : b+chunkSize]
			iGrad := input.Grad[b : b+chunkSize]

			softmax := output.Data[b : b+chunkSize]

			for i := 0; i < len(oGrad); i++ {
				g := oGrad[i]
				for j := 0; j < len(oGrad); j++ {
					if i == j {
						iGrad[j] += g * softmax[i] * (1 - softmax[i])
					} else {
						iGrad[j] += -g * softmax[i] * softmax[j]
					}
				}
			}
		}
	}

	return output
}
