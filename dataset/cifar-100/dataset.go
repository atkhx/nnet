package cifar_100

import (
	"fmt"
	"os"

	"github.com/atkhx/nnet/data"
	"github.com/pkg/errors"
)

func Open(filename string, rgb bool) (*dataset, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	imagesCount := len(b) / SampleSize

	var images []float64

	//nolint:gomnd
	if rgb {
		images = make([]float64, imagesCount*ImageSizeRGB)
		for i := 0; i < imagesCount; i++ {
			imageOffset := i * ImageSizeRGB
			sampleOffset := i * SampleSize
			for j := 0; j < ImageSizeRGB; j++ {
				images[imageOffset+j] = float64(b[sampleOffset+2+j]) / 255.0
			}
		}
	} else {
		images = make([]float64, imagesCount*ImageSizeGray)
		for i := 0; i < imagesCount; i++ {
			imageOffset := i * ImageSizeGray
			sampleOffset := i * SampleSize

			for j := 0; j < ImageSizeGray; j++ {
				R := float64(b[sampleOffset+2+j]) / 255.0
				G := float64(b[sampleOffset+2+j+ImageWidth]) / 255.0
				B := float64(b[sampleOffset+2+j+ImageWidth+ImageHeight]) / 255.0

				images[imageOffset+j] = (R + G + B) / 3
			}
		}
	}

	var classesIdx = make([]byte, imagesCount)
	var labelsIdx = make([]byte, imagesCount)
	for i := 0; i < imagesCount; i++ {
		classesIdx[i] = b[i*SampleSize]
		labelsIdx[i] = b[i*SampleSize+1]
	}

	res := &dataset{
		labels:       Labels,
		classes:      Classes,
		targets:      data.MustCompileOneHotVectors(len(Labels)),
		images:       images,
		samplesCount: imagesCount,
		labelsIdx:    labelsIdx,
		classesIdx:   classesIdx,
		rgb:          rgb,
	}

	return res, nil
}

type dataset struct {
	labels  []string
	classes []string
	targets []*data.Data
	images  []float64

	samplesCount int
	labelsIdx    []byte
	classesIdx   []byte

	rgb bool
}

func (d *dataset) GetSamplesCount() int {
	return d.samplesCount
}

func (d *dataset) GetLabels() []string {
	return d.labels
}

func (d *dataset) GetTargets() []*data.Data {
	return d.targets
}

func (d *dataset) GetLabel(index int) (string, error) {
	if index > -1 && index < len(d.labels) {
		return d.labels[index], nil
	}
	return "", ErrorIndexOutOfRange
}

func (d *dataset) GetTarget(index int) (*data.Data, error) {
	if index > -1 && index < len(d.targets) {
		return d.targets[index], nil
	}
	return nil, errors.Wrap(ErrorIndexOutOfRange, fmt.Sprintf("get target %d failed", index))
}

func (d *dataset) ReadSample(index int) (input, target *data.Data, err error) {
	label := d.labelsIdx[index]

	target, err = d.GetTarget(int(label))
	if err != nil {
		err = errors.Wrap(err, fmt.Sprintf("get target %d failed", label))
		return
	}

	input = &data.Data{}

	//nolint:gomnd
	if d.rgb {
		input.Init3DWithData(ImageWidth, ImageHeight, 3, d.images[index*ImageSizeRGB:(index+1)*ImageSizeRGB])
	} else {
		input.Init3DWithData(ImageWidth, ImageHeight, 1, d.images[index*ImageSizeGray:(index+1)*ImageSizeGray])
	}

	return
}
