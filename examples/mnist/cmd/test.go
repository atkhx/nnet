package main

import "fmt"

func calcSizes(iw, ih, fw, fh, padding, stride int) []int {
	ow := (iw-fw+2*padding)/stride + 1
	oh := (ih-fh+2*padding)/stride + 1
	return []int{ow, oh}
}

func main() {
	fmt.Println("28x28 3x3 padding 1", calcSizes(28, 28, 3, 3, 1, 1))
	fmt.Println("28x28 3x3 padding 0", calcSizes(28, 28, 3, 3, 0, 1))
}
