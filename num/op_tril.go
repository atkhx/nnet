package num

import (
	"runtime"
	"sync"
)

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

func (aData *Data) TriangleLowerMatrixMultiply(factor *Data) *Data { //nolint:gocyclo
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

	iW, iH := aData.Dims.W, aData.Dims.H
	fW, fH := factor.Dims.W, factor.Dims.H
	iWH := iW * iH
	fWH := fW * fH

	wg := sync.WaitGroup{}
	cn := make(chan struct{}, runtime.GOMAXPROCS(0))

	output.calcData = func() {
		var ozOffset, izOffset, fzOffset int

		wg.Add(oD)
		defer wg.Wait()

		for z := 0; z < oD; z++ {
			cn <- struct{}{}
			go func(ozOffset, izOffset, fzOffset int) {
				y := 0
				for oY := izOffset; oY < izOffset+iWH; oY += aData.Dims.W {
					y++
					iData := aData.Data[oY : oY+y]
					for oX := fzOffset; oX < fzOffset+fWH; oX += aData.Dims.W {
						fData := fTranspose.Data[oX : oX+y]

						v := 0.0
						for i, iV := range iData {
							v += iV * fData[i]
						}

						output.Data[ozOffset] = v
						ozOffset++
					}
				}
				<-cn
				wg.Done()
			}(ozOffset, izOffset, fzOffset)

			ozOffset += oW * oH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.calcData = func() {
		var ozOffset, izOffset, fzOffset int
		wg.Add(oD)
		defer wg.Wait()

		output.Data.Zero()
		for z := 0; z < oD; z++ {
			cn <- struct{}{}
			go func(aData, bData, oData Float64s) {
				mm_tr_lower(iW, aData, bData, oData)
				<-cn
				wg.Done()
			}(
				aData.Data[izOffset:izOffset+iWH],
				factor.Data[fzOffset:fzOffset+fWH],
				output.Data[ozOffset:ozOffset+(oW*oH)],
			)

			ozOffset += oW * oH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.calcGrad = func() {
		var ozOffset, izOffset, fzOffset int

		wg.Add(oD)
		defer wg.Wait()

		for z := 0; z < oD; z++ {
			cn <- struct{}{}
			go func(ozOffset, izOffset, fzOffset int) {
				y := 0
				for oY := izOffset; oY < izOffset+iWH; oY += aData.Dims.W {
					y++
					iData := aData.Data[oY : oY+y]
					iGrad := aData.Grad[oY : oY+y]

					for oX := fzOffset; oX < fzOffset+fWH; oX += aData.Dims.W {
						G := output.Grad[ozOffset]
						ozOffset++

						if G == 0 {
							continue
						}

						fData := fTranspose.Data[oX : oX+y]
						fGrad := fTranspose.Grad[oX : oX+y]

						for i, iV := range iData {
							fGrad[i] += G * iV
							iGrad[i] += G * fData[i]
						}
					}
				}

				<-cn
				wg.Done()
			}(ozOffset, izOffset, fzOffset)

			ozOffset += oW * oH
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
	cn := make(chan struct{}, runtime.GOMAXPROCS(0))

	output := aData.Copy()
	output.Data.Fill(0)
	output.calcData = func() {
		wg.Add(output.Dims.D)
		for z := 0; z < output.Dims.D; z++ {
			cn <- struct{}{}
			go func(z int) {
				for y := 0; y < output.Dims.H; y++ {
					c := z*WH + y*aData.Dims.W
					aData.Data[c:c+y+1].SoftmaxKTo(output.Data[c:c+y+1], k)
				}
				wg.Done()
				<-cn
			}(z)
		}
		wg.Wait()
	}

	output.calcGrad = func() {
		wg.Add(output.Dims.D)
		for z := 0; z < output.Dims.D; z++ {
			cn <- struct{}{}
			go func(z int) {
				for y := 0; y < output.Dims.H; y++ {
					c := z*WH + y*aData.Dims.W

					iGrad := aData.Grad[c : c+y+1]
					softmax := output.Data[c : c+y+1]

					for i, softmaxI := range softmax {
						if softmaxI == 0 {
							continue
						}

						sum := 0.0
						for j, softmaxJ := range softmax {
							if i == j {
								sum += (1 - softmaxJ) * output.Grad[c+j]
							} else {
								sum -= softmaxJ * output.Grad[c+j]
							}
						}

						iGrad[i] += softmaxI * k * sum
					}
				}
				wg.Done()
				<-cn
			}(z)
		}
		wg.Wait()
	}

	output.calcGrad = func() {
		wg.Add(output.Dims.D)
		for z := 0; z < output.Dims.D; z++ {
			cn <- struct{}{}
			go func(z int) {
				for y := 0; y < output.Dims.H; y++ {
					c := z*WH + y*aData.Dims.W

					iGrad := aData.Grad[c : c+y+1]
					softmax := output.Data[c : c+y+1]

					s := 0.0
					for i, softmaxI := range softmax {
						g := softmaxI * output.Grad[c+i] * k
						s += g
						iGrad[i] += g
					}

					for i, softmaxI := range softmax {
						iGrad[i] -= softmaxI * s
					}
				}
				wg.Done()
				<-cn
			}(z)
		}
		wg.Wait()
	}

	return output
}
