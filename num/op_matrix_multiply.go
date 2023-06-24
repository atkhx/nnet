package num

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

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

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{aData},
	}

	iWH := aData.Dims.W * aData.Dims.H
	fWH := factor.Dims.W * factor.Dims.H
	oWH := aData.Dims.H * factor.Dims.W

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
		// aData @ fData = oData
		AData := mat.NewDense(aData.Dims.H, aData.Dims.W, aData.Data[a.izOffset:a.izOffset+iWH])
		FData := mat.NewDense(factor.Dims.H, factor.Dims.W, factor.Data[a.fzOffset:a.fzOffset+fWH])
		OData := mat.NewDense(aData.Dims.H, factor.Dims.W, output.Data[a.offset:a.offset+oWH])

		OData.Mul(AData, FData)
		wg.Done()
	}

	backwardChan := make(chan backwardArgs, bufferSize)
	backwardFunc := func(a backwardArgs) {
		AData := mat.NewDense(aData.Dims.H, aData.Dims.W, aData.Data[a.izOffset:a.izOffset+iWH])
		AGrad := mat.NewDense(aData.Dims.H, aData.Dims.W, aData.Grad[a.izOffset:a.izOffset+iWH])

		FData := mat.NewDense(factor.Dims.H, factor.Dims.W, factor.Data[a.fzOffset:a.fzOffset+fWH])
		FGrad := mat.NewDense(factor.Dims.H, factor.Dims.W, factor.Grad[a.fzOffset:a.fzOffset+fWH])

		OGrad := mat.NewDense(aData.Dims.H, factor.Dims.W, output.Grad[a.offset:a.offset+oWH])

		// aGrad = oGrad @ fData_T
		// fGrad = aData_T @ oGrad
		AGrad.Mul(OGrad, FData.T())
		FGrad.Mul(AData.T(), OGrad)

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

func (aData *Data) MatrixMultiply2(factor *Data) *Data { //nolint:gocyclo
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
