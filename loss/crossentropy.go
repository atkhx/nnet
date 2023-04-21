package loss

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func CrossEntropy(target, actual num.Float64s) (loss float64) {
	actual = actual.Copy()
	actual.Softmax()

	for i, t := range target {
		actual[i] = -t * math.Log(actual[i])
	}
	return actual.Sum()
}

func CrossEntropyMean(bSize int, target, actual num.Float64s) (loss float64) {
	chunkSize := len(actual) / bSize
	actual = actual.Copy()

	for i := 0; i < len(actual); i += chunkSize {
		actual[i : i+chunkSize].Softmax()
	}

	for i, t := range target {
		actual[i] = -t * math.Log(actual[i])
	}

	return actual.Sum() / float64(bSize)
}

//
//func (m *Data) CrossEntropy(targets *Data) (outMatrix *Data) {
//	if !m.Data.IsDimensionsEqual(targets) {
//		panic(fmt.Sprintf(
//			"invalid targets dimensions: expected %v, actual %v",
//			m.GetDims(),
//			targets.GetDims(),
//		))
//	}
//
//	softmax := m.Data.Copy()
//	softmax.ScanRowsVolume(func(y, z int, f *Volume) {
//		f.Softmax()
//	})
//
//	logLikelihood := NewVolume(1, m.Data.H, m.Data.D)
//
//	targets.Data.ScanRows(func(y, z int, row []float64) {
//		for i, t := range row {
//			logLikelihood.PointAdd(0, y, z, -t*math.Log(softmax.At(i, y, z)))
//		}
//	})
//
//	return m.generate(logLikelihood, func() {
//		outMatrix.Grad.ScanRows(func(y, z int, f []float64) {
//			for x := 0; x < m.Data.W; x++ {
//				m.Grad.PointAdd(x, y, z, f[0]*(softmax.At(x, y, z)-targets.Data.At(x, y, z)))
//			}
//		})
//	})
//}
