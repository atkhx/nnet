package num

func max[T int | int64 | float64](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func mm(aW int, a, b, c Float64s) {
	aH := len(a) / aW
	fW := len(b) / aW

	for y := 0; y < aH; y++ {
		for x, v := range a[y*aW : y*aW+aW] {
			if v != 0 {
				axpyUnitary(v, b[x*fW:x*fW+fW], c[y*fW:y*fW+fW])
			}
		}
	}
}

func mmTB(aW int, a, b, c Float64s) {
	aH := len(a) / aW
	fW := len(b) / aW

	for y := 0; y < aH; y++ {
		a := a[y*aW : y*aW+aW]
		c := c[y*fW : y*fW+fW]

		for x, aV := range a {
			if aV == 0 {
				continue
			}

			for bY := range c {
				c[bY] += aV * b[bY*aW+x]
			}
		}
	}
}

func axpyUnitary(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

func transpose(aW, aH int, aData Float64s) Float64s {
	oData := aData.CopyZero()

	WH := aW * aH
	for d := 0; d < len(aData); d += WH {
		for y := 0; y < aH; y++ {
			for x := 0; x < aW; x++ {
				oData[d+x*aH+y] = aData[d+y*aW+x]
			}
		}
	}
	return oData
}

func GetMinMaxValues(data []float64) (min, max float64) {
	for i := 0; i < len(data); i++ {
		if i == 0 || min > data[i] {
			min = data[i]
		}
		if i == 0 || max < data[i] {
			max = data[i]
		}
	}
	return
}

func dot(iData, fData Float64s) (v float64) {
	for i, iV := range iData {
		v += iV * fData[i]
	}
	return v
}
