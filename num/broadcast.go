package num

type Steps struct {
	aW, aH, aD int
	bW, bH, bD int
}

func GetBroadCastStep(aValue, bValue int, dimension string) (aStep, bStep int) {
	aStep, bStep = 1, 1
	if aValue != bValue {
		switch {
		case aValue == 1:
			aStep = 0
		case bValue == 1:
			bStep = 0
		default:
			panic("A & B: " + dimension + " must be equal or one of them must be 1")
		}
	}
	return
}

func GetBroadCastSteps(aDims, bDims Dims) (steps Steps) {
	steps.aW, steps.bW = GetBroadCastStep(aDims.W, bDims.W, "width")
	steps.aH, steps.bH = GetBroadCastStep(aDims.H, bDims.H, "height")
	steps.aD, steps.bD = GetBroadCastStep(aDims.D, bDims.D, "depth")

	return
}

func BroadCast(aData, bData *Data) broadCastConfig {
	steps := GetBroadCastSteps(aData.Dims, bData.Dims)

	bc := broadCastConfig{
		axStep: steps.aW,
		ayStep: steps.aH * aData.Dims.W,
		azStep: steps.aD * aData.Dims.W * aData.Dims.H,

		bxStep: steps.bW,
		byStep: steps.bH * bData.Dims.W,
		bzStep: steps.bD * bData.Dims.W * bData.Dims.H,

		oDims: aData.Dims.GetMax(bData.Dims),
	}

	bc.coordsMap = prepareCoordsMap(bc)
	return bc
}

type broadCastConfig struct {
	axStep, ayStep, azStep int
	bxStep, byStep, bzStep int

	oDims     Dims
	coordsMap []bcCoordinate
}

type bcCoordinate struct {
	aOffset, bOffset int
}

func (cfg broadCastConfig) BroadCast(fn func(aOffset, bOffset, oOffset int)) {
	for offset, coords := range cfg.coordsMap {
		fn(coords.aOffset, coords.bOffset, offset)
	}
}

func prepareCoordsMap(cfg broadCastConfig) []bcCoordinate {
	var coordsMap []bcCoordinate

	offset := 0
	azOffset := 0
	bzOffset := 0
	for oZ := 0; oZ < cfg.oDims.D; oZ++ {

		ayOffset := 0
		byOffset := 0
		for oY := 0; oY < cfg.oDims.H; oY++ {

			axOffset := 0
			bxOffset := 0
			for oX := 0; oX < cfg.oDims.W; oX++ {
				coordsMap = append(coordsMap, bcCoordinate{
					aOffset: azOffset + ayOffset + axOffset,
					bOffset: bzOffset + byOffset + bxOffset,
				})

				offset++

				axOffset += cfg.axStep
				bxOffset += cfg.bxStep
			}

			ayOffset += cfg.ayStep
			byOffset += cfg.byStep
		}

		azOffset += cfg.azStep
		bzOffset += cfg.bzStep
	}

	return coordsMap
}
