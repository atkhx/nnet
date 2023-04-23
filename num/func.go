package num

import "fmt"

func GetRepeatedPosPairs(aLen, bLen int) [][2]int {
	mln := max(aLen, bLen)
	res := make([][2]int, mln)
	fi, bi := 0, 0
	for i := range res {
		res[i][0] = fi
		res[i][1] = bi

		fi++
		bi++

		if fi == aLen {
			fi = 0
		}

		if bi == bLen {
			bi = 0
		}
	}
	return res
}

func max[T int | float64](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func min[T int | float64](a, b T) T {
	if a < b {
		return a
	}
	return b
}

func Dot(a, b Float64s) (r float64) {
	for i, aV := range a {
		r += aV * b[i]
	}
	return
}

func RepeatDotTo(out, a, b Float64s) {
	if len(a) > len(b) {
		for i := range out {
			out[i] = Dot(a[i*len(b):(i+1)*len(b)], b)
		}
	} else {
		fmt.Println("len(a)", len(a))
		fmt.Println("len(b)", len(b))
		fmt.Println("len(o)", len(out))
		fmt.Println("neurons", len(out))
		for i := range out {
			out[i] = Dot(a, b[i*len(a):(i+1)*len(a)])
		}
	}
}
