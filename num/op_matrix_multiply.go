package num

func (input *Data) MatrixMultiply(factor *Data) *Data {
	if input.Dims.W != factor.Dims.H {
		panic("input width must be equal factor height")
	}

	izStep := 1
	fzStep := 1

	if input.Dims.D != factor.Dims.D {
		switch {
		case input.Dims.D == 1:
			izStep = 0
		case factor.Dims.D == 1:
			fzStep = 0
		default:
			panic("input's and factor's dept must be equal or one of them must be 1")
		}
	}

	oH := input.Dims.H
	oW := factor.Dims.W
	oD := input.Dims.D

	if factor.Dims.D > oD {
		oD = factor.Dims.D
	}

	fTranspose := factor.Transpose()

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{input, fTranspose},
	}

	iWH := input.Dims.W * input.Dims.H
	fWH := factor.Dims.W * factor.Dims.H

	output.calcData = func() {
		fTranspose.Forward()

		offset := 0

		izOffset := 0
		fzOffset := 0

		for z := 0; z < oD; z++ {
			for oY := izOffset; oY < izOffset+iWH; oY += input.Dims.W {
				iData := input.Data[oY : oY+input.Dims.W]

				for oX := fzOffset; oX < fzOffset+fWH; oX += input.Dims.W {
					fData := fTranspose.Data[oX : oX+input.Dims.W]

					v := 0.0
					for i, iV := range iData {
						v += iV * fData[i]
					}

					output.Data[offset] = v
					offset++
				}
			}

			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.calcGrad = func() {
		offset := 0

		izOffset := 0
		fzOffset := 0

		for z := 0; z < oD; z++ {
			for oY := izOffset; oY < izOffset+iWH; oY += input.Dims.W {
				iData := input.Data[oY : oY+input.Dims.W]
				iGrad := input.Grad[oY : oY+input.Dims.W]
				for oX := fzOffset; oX < fzOffset+fWH; oX += input.Dims.W {
					fData := fTranspose.Data[oX : oX+input.Dims.W]
					fGrad := fTranspose.Grad[oX : oX+input.Dims.W]

					G := output.Grad[offset]
					offset++

					for i, iV := range iData {
						fGrad[i] += G * iV
						iGrad[i] += G * fData[i]
					}
				}
			}

			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	return output
}
