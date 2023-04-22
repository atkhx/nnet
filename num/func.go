package num

func getRepeatedPosPairs(aLen, bLen int) [][2]int {
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
