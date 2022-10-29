package mnist

import (
	"io/ioutil"
)

const (
	labelsFileChunk  = 1
	labelsFileOffset = 2 * 4
)

func OpenLabelsFile(filename string) (*fileLabels, error) {
	res := &fileLabels{}
	err := res.open(filename)

	if err != nil {
		return nil, err
	}
	return res, nil
}

type fileLabels struct {
	data  []byte
	count int
}

func (f *fileLabels) open(filename string) error {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	f.data = b
	f.count = (len(b) - labelsFileOffset) / labelsFileChunk
	return nil
}

func (f *fileLabels) GetImagesCount() int {
	return f.count
}

func (f *fileLabels) ReadLabel(index int) (byte, error) {
	return f.data[int(labelsFileOffset+int64(index*labelsFileChunk))], nil
}
