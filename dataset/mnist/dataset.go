package mnist

import (
	"fmt"

	"github.com/atkhx/nnet/data"
	"github.com/pkg/errors"
)

var ErrorIndexOutOfRange = errors.New("index out of range")

func New(imagesFile *fileImages, labelsFile *fileLabels) (*dataset, error) {
	if imagesFile.GetImagesCount() != labelsFile.GetImagesCount() {
		return nil, errors.New("images count <> labels count")
	}

	return &dataset{
		labels:      []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
		targets:     data.MustCompileOneHotVectors(10),
		imagesFile:  imagesFile,
		labelsFile:  labelsFile,
		imagesCount: imagesFile.GetImagesCount(),
	}, nil
}

type dataset struct {
	labels  []string
	targets []*data.Data

	imagesFile  *fileImages
	labelsFile  *fileLabels
	imagesCount int
}

func (d *dataset) GetSamplesCount() int {
	return d.imagesCount
}

func (d *dataset) GetLabels() []string {
	return d.labels
}

func (d *dataset) GetLabel(index int) (string, error) {
	if index > -1 && index < len(d.labels) {
		return d.labels[index], nil
	}
	return "", ErrorIndexOutOfRange
}

func (d *dataset) GetTargets() []*data.Data {
	return d.targets
}

func (d *dataset) GetTarget(index int) (*data.Data, error) {
	if index > -1 && index < len(d.targets) {
		return d.targets[index], nil
	}
	return nil, ErrorIndexOutOfRange
}

func (d *dataset) ReadBytes(index int) (image []float64, label byte, err error) {
	if index < 0 || index > d.imagesCount {
		return nil, 0, ErrorIndexOutOfRange
	}

	if image, err = d.imagesFile.ReadImage(index); err != nil {
		return
	}

	if label, err = d.labelsFile.ReadLabel(index); err != nil {
		return
	}
	return
}

func (d *dataset) ReadSample(index int) (input, target *data.Data, err error) {
	image, label, err := d.ReadBytes(index)
	if err != nil {
		return
	}

	target, err = d.GetTarget(int(label))
	if err != nil {
		err = errors.Wrap(err, fmt.Sprintf("get target %d failed", label))
		return
	}

	input = &data.Data{}
	input.InitMatrixWithData(ImageWidth, ImageHeight, image)
	return
}
