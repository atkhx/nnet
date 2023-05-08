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

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{input, factor},
	}

	iWH := input.Dims.W * input.Dims.H
	fWH := factor.Dims.W * factor.Dims.H

	output.calcData = func() {
		offset := 0

		izOffset := 0
		fzOffset := 0

		for z := 0; z < oD; z++ {
			for oY := 0; oY < oH; oY++ {
				for oX := 0; oX < oW; oX++ {

					v := 0.0
					for i, iV := range input.Data[izOffset+oY*input.Dims.W : izOffset+(oY+1)*input.Dims.W] {
						v += iV * factor.Data[fzOffset+i*factor.Dims.W+oX]
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
			for oY := 0; oY < oH; oY++ {
				for oX := 0; oX < oW; oX++ {
					G := output.Grad[offset]
					offset++

					iData := input.Data[izOffset+oY*input.Dims.W : izOffset+(oY+1)*input.Dims.W]
					iGrad := input.Grad[izOffset+oY*input.Dims.W : izOffset+(oY+1)*input.Dims.W]

					for i, iV := range iData {
						factor.Grad[fzOffset+i*factor.Dims.W+oX] += G * iV
						iGrad[i] += G * factor.Data[fzOffset+i*factor.Dims.W+oX]
					}
				}
			}

			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	return output
}
