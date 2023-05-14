package num

func (input *Data) Add(bData *Data) *Data {
	oDims := input.Dims.GetDimsByMax(bData.Dims)
	steps := input.Dims.GetBroadCastSteps(bData.Dims)

	output := New(oDims, input, bData)

	izStep := steps.aD * input.Dims.W * input.Dims.H
	iyStep := steps.aH * input.Dims.W

	bzStep := steps.bD * bData.Dims.W * bData.Dims.H
	byStep := steps.bH * bData.Dims.W

	output.SetOperation("add")
	output.calcData = func() {
		offset := 0

		izOffset := 0
		bzOffset := 0
		for oZ := 0; oZ < oDims.D; oZ++ {

			iyOffset := 0
			byOffset := 0
			for oY := 0; oY < oDims.H; oY++ {

				ixOffset := 0
				bxOffset := 0
				for oX := 0; oX < oDims.W; oX++ {
					iV := input.Data[izOffset+iyOffset+ixOffset]
					bV := bData.Data[bzOffset+byOffset+bxOffset]

					output.Data[offset] = iV + bV
					offset++

					ixOffset += steps.aW
					bxOffset += steps.bW
				}

				iyOffset += iyStep
				byOffset += byStep
			}

			izOffset += izStep
			bzOffset += bzStep
		}
	}

	output.calcGrad = func() {
		offset := 0

		izOffset := 0
		bzOffset := 0
		for oZ := 0; oZ < oDims.D; oZ++ {

			iyOffset := 0
			byOffset := 0
			for oY := 0; oY < oDims.H; oY++ {

				ixOffset := 0
				bxOffset := 0
				for oX := 0; oX < oDims.W; oX++ {
					input.Grad[izOffset+iyOffset+ixOffset] += output.Grad[offset]
					bData.Grad[bzOffset+byOffset+bxOffset] += output.Grad[offset]

					offset++

					ixOffset += steps.aW
					bxOffset += steps.bW
				}

				iyOffset += iyStep
				byOffset += byStep
			}

			izOffset += izStep
			bzOffset += bzStep
		}
	}

	return output
}
