package num

import (
	"math"
	"runtime"
	"sync"
)

func (aData *Data) CrossEntropyPos(targets *Data) *Data {
	if targets.Dims.W != 1 {
		panic("target width must be equal 1")
	}

	if targets.Dims.H != aData.Dims.H {
		panic("target height must be equal aData height")
	}

	if targets.Dims.D != aData.Dims.D {
		panic("target depth must be equal aData depth")
	}

	oDims := targets.Dims
	chunkSize := aData.Dims.W

	// just buffer to avoid memory allocations
	softmax := aData.Data.CopyZero()

	wg := sync.WaitGroup{}
	cn := make(chan struct{}, runtime.GOMAXPROCS(0))

	output := New(oDims, aData)
	output.calcData = func() {
		softmax.CopyFrom(aData.Data)
		wg.Add(len(softmax) / chunkSize)
		for i := 0; i < len(softmax); i += chunkSize {
			cn <- struct{}{}
			go func(i int) {
				softmax[i : i+chunkSize].Softmax()

				<-cn
				wg.Done()
			}(i)
		}
		wg.Wait()

		for rowIdx, correctIdx := range targets.Data {
			for i := 0; i < chunkSize; i++ {
				if i == int(correctIdx) {
					output.Data[rowIdx] = -math.Log(softmax[rowIdx*chunkSize+i])
				}
			}
		}
	}

	output.calcGrad = func() {
		for rowIdx, correctIdx := range targets.Data {
			for i := 0; i < chunkSize; i++ {
				j := rowIdx*chunkSize + i
				t := 0.0
				if i == int(correctIdx) {
					t = 1.0
				}
				aData.Grad[j] += output.Grad[0] * (softmax[j] - t)
			}
		}
	}

	return output
}

func (aData *Data) CrossEntropy(targets *Data) *Data {
	aData.Dims.MustBeEqual(targets.Dims)
	oDims := aData.Dims
	oDims.W = 1
	chunkSize := aData.Dims.W

	// just buffer to avoid memory allocations
	softmax := aData.Data.CopyZero()
	logLikelihood := aData.Data.CopyZero()

	output := New(oDims, aData)
	output.calcData = func() {
		softmax.CopyFrom(aData.Data)
		for i := 0; i < len(softmax); i += chunkSize {
			softmax[i : i+chunkSize].Softmax()
		}

		for i, t := range targets.Data {
			logLikelihood[i] = -t * math.Log(softmax[i])
		}

		for i := range output.Data {
			output.Data[i] = logLikelihood[i*chunkSize : (i+1)*chunkSize].Sum()
		}
	}
	output.calcGrad = func() {
		for i, t := range targets.Data {
			aData.Grad[i] += softmax[i] - t
		}
	}

	return output
}
