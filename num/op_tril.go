package num

import (
	"runtime"
	"sync"
)

func (aData *Data) TriangleLowerSoftmax() *Data {
	WH := aData.Dims.W * aData.Dims.W

	wg := sync.WaitGroup{}
	cn := make(chan struct{}, runtime.GOMAXPROCS(0))

	output := aData.NewLinkedCopy()
	output.Data.Zero()
	output.calcData = func() {
		wg.Add(output.Dims.D)
		for z := 0; z < output.Dims.D; z++ {
			cn <- struct{}{}
			go func(z int) {
				for y := 0; y < output.Dims.H; y++ {
					c := z*WH + y*aData.Dims.W
					aData.Data[c : c+y+1].SoftmaxTo(output.Data[c : c+y+1])
				}
				<-cn
				wg.Done()
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
						g := softmaxI * output.Grad[c+i]
						s += g
						iGrad[i] += g
					}

					for i, softmaxI := range softmax {
						iGrad[i] -= softmaxI * s
					}
				}
				<-cn
				wg.Done()
			}(z)
		}
		wg.Wait()
	}

	return output
}
