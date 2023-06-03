package num

import "fmt"

func FromImages(w, h, d int, images ...Float64s) (outData *Data) {
	if len(images) < 1 {
		panic("images data is required")
	}

	if len(images[0]) != w*h*d {
		panic(fmt.Sprintf("image data length is not equal %d * %d * %d", w, h, d))
	}

	whd := w * h * d
	data := NewFloat64s(whd * len(images))

	offset := 0
	for _, image := range images {
		copy(data[offset:offset+whd], image)
		offset += whd
	}

	return NewWithValues(NewDims(w*h, d, len(images)), data)
}
