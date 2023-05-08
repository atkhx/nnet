package num

func (input *Data) GetEmbedded(tokens *Data) *Data {
	// alphabetSize := input.Dims.H
	featuresCount := input.Dims.W

	contextSize := tokens.Dims.W
	tokensCount := tokens.Dims.H

	output := New(NewDims(
		featuresCount,
		contextSize,
		tokensCount,
	), input)

	output.calcData = func() {
		for i, p := range tokens.Data.ToInt() {
			features := input.Data[p*featuresCount : (p+1)*featuresCount]
			outBuffer := output.Data[i*featuresCount : (i+1)*featuresCount]

			copy(outBuffer, features)
		}
	}

	output.calcGrad = func() {
		for i, p := range tokens.Data.ToInt() {
			wGrads := input.Grad[p*featuresCount : (p+1)*featuresCount]
			oGrads := output.Grad[i*featuresCount : (i+1)*featuresCount]

			for j, g := range oGrads {
				wGrads[j] += g
			}
		}
	}

	return output
}

func (input *Data) GetEmbeddedPos(tokens *Data) *Data {
	featuresCount := input.Dims.W

	contextSize := tokens.Dims.W
	tokensCount := tokens.Dims.H

	output := New(NewDims(
		featuresCount,
		contextSize,
		tokensCount,
	), input)

	output.calcData = func() {
		p := 0
		for i := range tokens.Data {
			features := input.Data[p*featuresCount : (p+1)*featuresCount]
			outBuffer := output.Data[i*featuresCount : (i+1)*featuresCount]
			copy(outBuffer, features)

			p++
			if p == contextSize {
				p = 0
			}
		}
	}

	output.calcGrad = func() {
		p := 0
		for i := range tokens.Data {
			wGrads := input.Grad[p*featuresCount : (p+1)*featuresCount]
			oGrads := output.Grad[i*featuresCount : (i+1)*featuresCount]

			for j, g := range oGrads {
				wGrads[j] += g
			}

			p++
			if p == contextSize {
				p = 0
			}
		}
	}

	return output
}
