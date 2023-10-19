package broadcast

import "github.com/atkhx/nnet/num"

func MakeSteps(aDims, bDims num.Dims) (steps Steps) {
	steps.AWMul, steps.BWMul = getStepMultipliers(aDims.W, bDims.W, "width")
	steps.AHMul, steps.BHMul = getStepMultipliers(aDims.H, bDims.H, "height")
	steps.ADMul, steps.BDMul = getStepMultipliers(aDims.D, bDims.D, "depth")

	steps.AXStep = steps.AWMul
	steps.AYStep = steps.AHMul * aDims.W
	steps.AZStep = steps.ADMul * aDims.W * aDims.H

	steps.BXStep = steps.BWMul
	steps.BYStep = steps.BHMul * bDims.W
	steps.BZStep = steps.BDMul * bDims.W * bDims.H
	return
}

type Steps struct {
	AWMul, AHMul, ADMul int
	BWMul, BHMul, BDMul int

	AXStep, AYStep, AZStep int
	BXStep, BYStep, BZStep int
}

func getStepMultipliers(aValue, bValue int, dimension string) (aMul, bMul int) {
	aMul, bMul = 1, 1
	if aValue != bValue {
		switch {
		case aValue == 1:
			aMul = 0
		case bValue == 1:
			bMul = 0
		default:
			panic("A & B: " + dimension + " must be equal or one of them must be 1")
		}
	}
	return
}
