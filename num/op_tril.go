package num

import "sync"

func (aData *Data) TriangleLower(zeroVal float64) *Data {
	WH := aData.Dims.W * aData.Dims.W

	output := aData.Copy()
	output.calcData = func() {
		output.Data.Fill(zeroVal)
		for z := 0; z < output.Dims.D; z++ {
			for y := 0; y < output.Dims.H; y++ {
				for x := 0; x <= y; x++ {
					coordinate := z*WH + y*aData.Dims.W + x
					output.Data[coordinate] = aData.Data[coordinate]
				}
			}
		}
	}

	output.calcGrad = func() {
		for z := 0; z < output.Dims.D; z++ {
			for y := 0; y < output.Dims.H; y++ {
				for x := 0; x <= y; x++ {
					coordinate := z*WH + y*aData.Dims.W + x
					aData.Grad[coordinate] += output.Grad[coordinate]
				}
			}
		}
	}
	return output
}

func (aData *Data) TriangleLowerMatrixMultiply(factor *Data) *Data {
	if aData.Dims.W != factor.Dims.H {
		panic("aData width must be equal factor height")
	}

	izStep := 1
	fzStep := 1

	if aData.Dims.D != factor.Dims.D {
		switch {
		case aData.Dims.D == 1:
			izStep = 0
		case factor.Dims.D == 1:
			fzStep = 0
		default:
			panic("aData's and factor's dept must be equal or one of them must be 1")
		}
	}

	oH := aData.Dims.H
	oW := factor.Dims.W
	oD := aData.Dims.D

	if factor.Dims.D > oD {
		oD = factor.Dims.D
	}

	fTranspose := factor.Transpose()

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{aData, fTranspose},
	}

	iWH := aData.Dims.W * aData.Dims.H
	fWH := factor.Dims.W * factor.Dims.H

	wg := sync.WaitGroup{}

	type forwardArgs struct {
		offset, izOffset, fzOffset int
	}

	type backwardArgs struct {
		offset, izOffset, fzOffset int
	}

	bufferSize := 64
	forwardChan := make(chan forwardArgs, bufferSize)
	forwardFunc := func(a forwardArgs) {
		y := 0
		for oY := a.izOffset; oY < a.izOffset+iWH; oY += aData.Dims.W {
			y++
			//iData := aData.Data[oY : oY+aData.Dims.W]
			iData := aData.Data[oY : oY+y]
			for oX := a.fzOffset; oX < a.fzOffset+fWH; oX += aData.Dims.W {
				//fData := fTranspose.Data[oX : oX+aData.Dims.W]
				fData := fTranspose.Data[oX : oX+y]

				v := 0.0
				for i, iV := range iData {
					v += iV * fData[i]
				}

				output.Data[a.offset] = v
				a.offset++
			}
		}

		wg.Done()
	}

	backwardChan := make(chan backwardArgs, bufferSize)
	backwardFunc := func(a backwardArgs) {
		y := 0
		for oY := a.izOffset; oY < a.izOffset+iWH; oY += aData.Dims.W {
			y++
			iData := aData.Data[oY : oY+y]
			iGrad := aData.Grad[oY : oY+y]
			//iData := aData.Data[oY : oY+aData.Dims.W]
			//iGrad := aData.Grad[oY : oY+aData.Dims.W]

			for oX := a.fzOffset; oX < a.fzOffset+fWH; oX += aData.Dims.W {
				fData := fTranspose.Data[oX : oX+y]
				fGrad := fTranspose.Grad[oX : oX+y]

				G := output.Grad[a.offset]
				a.offset++

				for i, iV := range iData {
					fGrad[i] += G * iV
					iGrad[i] += G * fData[i]
				}
			}
		}

		wg.Done()
	}

	for i := 0; i < bufferSize; i++ {
		go func() {
			for args := range forwardChan {
				forwardFunc(args)
			}
		}()

		go func() {
			for args := range backwardChan {
				backwardFunc(args)
			}
		}()
	}

	output.calcData = func() {
		fTranspose.Forward()

		offset := 0
		izOffset := 0
		fzOffset := 0

		wg.Add(oD)
		for z := 0; z < oD; z++ {
			forwardChan <- forwardArgs{offset, izOffset, fzOffset}

			offset += oW * oH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}

		wg.Wait()
	}

	output.calcGrad = func() {
		offset := 0
		izOffset := 0
		fzOffset := 0

		wg.Add(oD)
		for z := 0; z < oD; z++ {
			backwardChan <- backwardArgs{offset, izOffset, fzOffset}

			offset += oW * oH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}

		wg.Wait()

		fTranspose.Backward()
	}

	return output
}

func (aData *Data) TriangleLowerMatrixMultiply2(factor *Data) *Data {
	if aData.Dims.W != factor.Dims.H {
		panic("input width must be equal factor height")
	}

	izStep := 1
	fzStep := 1

	if aData.Dims.D != factor.Dims.D {
		switch {
		case aData.Dims.D == 1:
			izStep = 0
		case factor.Dims.D == 1:
			fzStep = 0
		default:
			panic("input's and factor's dept must be equal or one of them must be 1")
		}
	}

	oH := aData.Dims.H
	oW := factor.Dims.W
	oD := aData.Dims.D

	if factor.Dims.D > oD {
		oD = factor.Dims.D
	}

	fTranspose := factor.Transpose()

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{aData, fTranspose},
	}

	iWH := aData.Dims.W * aData.Dims.H
	fWH := factor.Dims.W * factor.Dims.H

	output.calcData = func() {
		fTranspose.Forward()

		offset := 0

		izOffset := 0
		fzOffset := 0

		for z := 0; z < oD; z++ {
			y := 0
			for oY := izOffset; oY < izOffset+iWH; oY += aData.Dims.W {
				y++
				//iData := aData.Data[oY : oY+aData.Dims.W]
				iData := aData.Data[oY : oY+y]

				for oX := fzOffset; oX < fzOffset+fWH; oX += aData.Dims.W {
					//fData := fTranspose.Data[oX : oX+aData.Dims.W]
					fData := fTranspose.Data[oX : oX+y]

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
			y := 0
			for oY := izOffset; oY < izOffset+iWH; oY += aData.Dims.W {
				y++
				iData := aData.Data[oY : oY+y]
				iGrad := aData.Grad[oY : oY+y]
				//iData := aData.Data[oY : oY+aData.Dims.W]
				//iGrad := aData.Grad[oY : oY+aData.Dims.W]
				for oX := fzOffset; oX < fzOffset+fWH; oX += aData.Dims.W {
					fData := fTranspose.Data[oX : oX+y]
					fGrad := fTranspose.Grad[oX : oX+y]
					//fData := fTranspose.Data[oX : oX+aData.Dims.W]
					//fGrad := fTranspose.Grad[oX : oX+aData.Dims.W]

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

func (aData *Data) TriangleLowerSoftmax(k float64) *Data {
	if k == 0 {
		k = 1.0
	}
	WH := aData.Dims.W * aData.Dims.W

	wg := sync.WaitGroup{}

	output := aData.Copy()
	output.Data.Fill(0)
	output.calcData = func() {
		wg.Add(output.Dims.D)
		for z := 0; z < output.Dims.D; z++ {
			go func(z int) {
				for y := 0; y < output.Dims.H; y++ {
					c := z*WH + y*aData.Dims.W
					aData.Data[c:c+y+1].SoftmaxKTo(output.Data[c:c+y+1], k)
				}
				wg.Done()
			}(z)
		}
		wg.Wait()
		//
		//for z := 0; z < output.Dims.D; z++ {
		//	for y := 0; y < output.Dims.H; y++ {
		//		c := z*WH + y*aData.Dims.W
		//		aData.Data[c:c+y+1].SoftmaxKTo(output.Data[c:c+y+1], k)
		//	}
		//}
	}

	output.calcGrad = func() {
		wg.Add(output.Dims.D)
		for z := 0; z < output.Dims.D; z++ {
			go func(z int) {
				for y := 0; y < output.Dims.H; y++ {
					c := z*WH + y*aData.Dims.W

					iGrad := aData.Grad[c : c+y+1]
					softmax := output.Data[c : c+y+1]

					for i, softmaxI := range softmax {
						gI := k * output.Grad[c+i] * softmaxI
						for j, softmaxJ := range softmax {
							if i == j {
								iGrad[j] += gI * (1 - softmaxI)
							} else {
								iGrad[j] -= gI * softmaxJ
							}
						}
					}
				}
				wg.Done()
			}(z)
		}
		wg.Wait()

		//for z := 0; z < output.Dims.D; z++ {
		//	for y := 0; y < output.Dims.H; y++ {
		//		c := z*WH + y*aData.Dims.W
		//
		//		iGrad := aData.Grad[c : c+y+1]
		//		softmax := output.Data[c : c+y+1]
		//
		//		for i, softmaxI := range softmax {
		//			gI := k * output.Grad[c+i] * softmaxI
		//			for j, softmaxJ := range softmax {
		//				if i == j {
		//					iGrad[j] += gI * (1 - softmaxI)
		//				} else {
		//					iGrad[j] -= gI * softmaxJ
		//				}
		//			}
		//		}
		//	}
		//}
	}
	return output
}
