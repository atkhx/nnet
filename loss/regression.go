package loss

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func Regression(target, actual num.Float64s, bSize int) (lossMean float64) {
	for i, t := range target {
		lossMean += math.Pow(actual[i]-t, 2)
	}
	return 0.5 * lossMean / float64(bSize)
}

func RegressionBackward(target, actual num.Float64s, bSize int) (oGrads num.Float64s) {
	oGrads = actual.Copy()
	oGrads.AddWeighted(target, -1.0)
	oGrads.MulScalar(1.0 / float64(bSize))
	return oGrads
}
