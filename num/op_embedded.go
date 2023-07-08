package num

func (aData *Data) GetEmbeddings(tEmbeddings, pEmbeddings *Data) *Data {
	if tEmbeddings.Dims.W != pEmbeddings.Dims.W {
		panic("features count must be equal")
	}

	featuresCount := tEmbeddings.Dims.W

	contextSize := aData.Dims.W
	tokensCount := aData.Dims.H

	output := New(NewDims(
		featuresCount,
		contextSize,
		tokensCount,
	), tEmbeddings)
	//), tEmbeddings, pEmbeddings)

	output.calcData = func() {
		p := 0
		for i, s := range aData.Data.ToInt() {
			tFeatures := tEmbeddings.Data[s*featuresCount : (s+1)*featuresCount]
			pFeatures := pEmbeddings.Data[p*featuresCount : (p+1)*featuresCount]

			outBuffer := output.Data[i*featuresCount : (i+1)*featuresCount]
			outBuffer.CopyFrom(tFeatures)
			outBuffer.Add(pFeatures)

			p++
			if p == contextSize {
				p = 0
			}
		}
	}
	output.calcGrad = func() {
		//p := 0
		for i, s := range aData.Data.ToInt() {
			tGrads := tEmbeddings.Grad[s*featuresCount : (s+1)*featuresCount]
			tGrads.Add(output.Grad[i*featuresCount : (i+1)*featuresCount])
			//pGrads := pEmbeddings.Grad[p*featuresCount : (p+1)*featuresCount]

			//for j, g := range output.Grad[i*featuresCount : (i+1)*featuresCount] {
			//	tGrads[j] += g
			//pGrads[j] += g
			//}

			//p++
			//if p == contextSize {
			//	p = 0
			//}
		}
	}
	return output
}
