package broadcast

import "github.com/atkhx/nnet/num"

type Offsets struct {
	AOffset, BOffset, OOffset int
}

func MakeOffsets(steps Steps, outDims num.Dims) []Offsets {
	var coordsMap []Offsets

	oiOffset := 0
	azOffset := 0
	bzOffset := 0

	for oZ := 0; oZ < outDims.D; oZ++ {

		ayOffset := 0
		byOffset := 0
		for oY := 0; oY < outDims.H; oY++ {

			axOffset := 0
			bxOffset := 0
			for oX := 0; oX < outDims.W; oX++ {
				coordsMap = append(coordsMap, Offsets{
					AOffset: azOffset + ayOffset + axOffset,
					BOffset: bzOffset + byOffset + bxOffset,
					OOffset: oiOffset,
				})

				oiOffset++

				axOffset += steps.AXStep
				bxOffset += steps.BXStep
			}

			ayOffset += steps.AYStep
			byOffset += steps.BYStep
		}

		azOffset += steps.AZStep
		bzOffset += steps.BZStep
	}

	return coordsMap
}
