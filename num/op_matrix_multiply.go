package num

import "sync"

func (aData *Data) MatrixMultiply2(factor *Data) *Data {
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

	output.calcData = func() {
		offset := 0

		izOffset := 0
		fzOffset := 0

		var iI int
		var v float64

		for z := 0; z < oD; z++ {
			fData := fTranspose.Data[fzOffset : fzOffset+fWH]

			for oY := izOffset; oY < izOffset+iWH; oY += aData.Dims.W {
				iData := aData.Data[oY : oY+aData.Dims.W]

				for _, fV := range fData {
					v += iData[iI] * fV

					if iI++; iI == aData.Dims.W {
						output.Data[offset] = v
						offset++

						v = 0
						iI = 0
					}
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
			for oY := izOffset; oY < izOffset+iWH; oY += aData.Dims.W {
				iData := aData.Data[oY : oY+aData.Dims.W]
				iGrad := aData.Grad[oY : oY+aData.Dims.W]

				for oX := fzOffset; oX < fzOffset+fWH; oX += aData.Dims.W {
					fData := fTranspose.Data[oX : oX+aData.Dims.W]
					fGrad := fTranspose.Grad[oX : oX+aData.Dims.W]

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

func (aData *Data) MatrixMultiply(factor *Data) *Data { //nolint:gocyclo
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
		for oY := a.izOffset; oY < a.izOffset+iWH; oY += aData.Dims.W {
			iData := aData.Data[oY : oY+aData.Dims.W]
			for oX := a.fzOffset; oX < a.fzOffset+fWH; oX += aData.Dims.W {
				fData := fTranspose.Data[oX : oX+aData.Dims.W]

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
		for oY := a.izOffset; oY < a.izOffset+iWH; oY += aData.Dims.W {
			iData := aData.Data[oY : oY+aData.Dims.W]
			iGrad := aData.Grad[oY : oY+aData.Dims.W]

			for oX := a.fzOffset; oX < a.fzOffset+fWH; oX += aData.Dims.W {
				fData := fTranspose.Data[oX : oX+aData.Dims.W]
				fGrad := fTranspose.Grad[oX : oX+aData.Dims.W]

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
		//fTranspose.Forward()

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

		//fTranspose.Backward()
	}

	return output
}

func (aData *Data) MatrixMultiplyTransposed(fTranspose *Data) *Data { //nolint:gocyclo
	if aData.Dims.W != fTranspose.Dims.W {
		panic("aData width must be equal factor height")
	}

	izStep := 1
	fzStep := 1

	if aData.Dims.D != fTranspose.Dims.D {
		switch {
		case aData.Dims.D == 1:
			izStep = 0
		case fTranspose.Dims.D == 1:
			fzStep = 0
		default:
			panic("aData's and factor's dept must be equal or one of them must be 1")
		}
	}

	oH := aData.Dims.H
	oW := fTranspose.Dims.H
	oD := aData.Dims.D

	if fTranspose.Dims.D > oD {
		oD = fTranspose.Dims.D
	}

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{aData, fTranspose},
	}

	iWH := aData.Dims.W * aData.Dims.H
	fWH := fTranspose.Dims.W * fTranspose.Dims.H

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
		for oY := a.izOffset; oY < a.izOffset+iWH; oY += aData.Dims.W {
			iData := aData.Data[oY : oY+aData.Dims.W]
			for oX := a.fzOffset; oX < a.fzOffset+fWH; oX += aData.Dims.W {
				fData := fTranspose.Data[oX : oX+aData.Dims.W]

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
		for oY := a.izOffset; oY < a.izOffset+iWH; oY += aData.Dims.W {
			iData := aData.Data[oY : oY+aData.Dims.W]
			iGrad := aData.Grad[oY : oY+aData.Dims.W]

			for oX := a.fzOffset; oX < a.fzOffset+fWH; oX += aData.Dims.W {
				fData := fTranspose.Data[oX : oX+aData.Dims.W]
				fGrad := fTranspose.Grad[oX : oX+aData.Dims.W]

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
	}

	return output
}
