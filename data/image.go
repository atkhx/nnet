package data

import "fmt"

// we have
// - floats as image data
// - image width
// - image height

// we want matrix where image data is separately by channels stretched horizontally

func FromImages(w, h, d int, images ...[]float64) (outData *Data) {
	if len(images) < 1 {
		panic("images data is required")
	}

	if len(images[0]) != w*h*d {
		panic(fmt.Sprintf("image data length is not equal %d * %d * %d", w, h, d))
	}

	data := make([]float64, 0, w*h*d*len(images))
	for _, image := range images {
		data = append(data, image...)
	}
	return WrapData(w*h, d, len(images), data)
}
