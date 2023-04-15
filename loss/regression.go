package loss

import "math"

func Regression(target, actual []float64) (loss float64) {
	for i, t := range target {
		loss += math.Pow(actual[i]-t, 2)
	}
	return loss * 0.5
}
