package mnist

import (
	"os"
)

const (
	imagesFileChunk  = ImageWidth * ImageHeight
	imagesFileOffset = 4 * 4
)

func OpenImagesFile(filename string) (*fileImages, error) {
	res := &fileImages{}
	err := res.open(filename)

	if err != nil {
		return nil, err
	}
	return res, nil
}

type fileImages struct {
	data  []float64
	count int
}

func (f *fileImages) open(filename string) error {
	b, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	d := make([]float64, len(b))
	for i := 0; i < len(d); i++ {
		//nolint:gomnd
		d[i] = float64(b[i]) / 255.0
	}

	f.data = d
	f.count = (len(b) - imagesFileOffset) / imagesFileChunk
	return nil
}

func (f *fileImages) GetImagesCount() int {
	return f.count
}

func (f *fileImages) ReadImage(index int) ([]float64, error) {
	imageOffset := int(imagesFileOffset + int64(index*imagesFileChunk))
	return f.data[imageOffset : imageOffset+imagesFileChunk], nil
}
