package mnist

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"

	"github.com/atkhx/nnet/num"
)

var ErrorIndexOutOfRange = errors.New("index out of range")

func CreateTrainingDataset(datasetPath string) (*dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName)
	labelsFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainLabelsFileName)

	imagesFile, err := OpenImagesFile(imagesFileName)
	if err != nil {
		return nil, fmt.Errorf("open images file: %w", err)
	}

	labelsFile, err := OpenLabelsFile(labelsFileName)
	if err != nil {
		return nil, fmt.Errorf("open labels file: %w", err)
	}

	result, err := New(imagesFile, labelsFile)
	if err != nil {
		return nil, fmt.Errorf("create dataset: %w", err)
	}
	return result, nil
}

func CreateTestingDataset(datasetPath string) (*dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestImagesFileName)
	labelsFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestLabelsFileName)

	imagesFile, err := OpenImagesFile(imagesFileName)
	if err != nil {
		return nil, fmt.Errorf("open images file: %w", err)
	}

	labelsFile, err := OpenLabelsFile(labelsFileName)
	if err != nil {
		return nil, fmt.Errorf("open labels file: %w", err)
	}

	result, err := New(imagesFile, labelsFile)
	if err != nil {
		return nil, fmt.Errorf("create dataset: %w", err)
	}
	return result, nil
}

func New(imagesFile *fileImages, labelsFile *fileLabels) (*dataset, error) {
	if imagesFile.GetImagesCount() != labelsFile.GetImagesCount() {
		return nil, errors.New("images count not equals labels count")
	}

	return &dataset{
		labels:      labels,
		targets:     NewSeparateOneHotVectors(len(labels)),
		imagesFile:  imagesFile,
		labelsFile:  labelsFile,
		imagesCount: imagesFile.GetImagesCount(),
	}, nil
}

type dataset struct {
	labels  []string
	targets []*num.Data

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

func (d *dataset) GetTargets() []*num.Data {
	return d.targets
}

func (d *dataset) GetTarget(index int) (*num.Data, error) {
	if index > -1 && index < len(d.targets) {
		return d.targets[index], nil
	}
	return nil, ErrorIndexOutOfRange
}

func (d *dataset) GetTargetsByIndexes(index ...int) (*num.Data, error) {
	return NewOneHotVectors(len(d.targets), index...), nil
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

func (d *dataset) ReadSample(index int) (input, target *num.Data, err error) {
	image, label, err := d.ReadBytes(index)
	if err != nil {
		return
	}

	target, err = d.GetTarget(int(label))
	if err != nil {
		err = fmt.Errorf("get target %d: %w", label, err)
		return
	}

	input = FromImages(ImageWidth, ImageHeight, ImageDepth, image)
	return
}

func (d *dataset) ReadRandomSampleBatch(batchSize int) (input, target *num.Data, err error) {
	var images [][]float64
	var labels []int

	for i := 0; i < batchSize; i++ {
		image, label, e := d.ReadBytes(rand.Intn(d.GetSamplesCount())) //nolint:gosec
		if e != nil {
			err = e
			return
		}

		images = append(images, image)
		labels = append(labels, int(label))
	}

	input = FromImages(ImageWidth, ImageHeight, ImageDepth, images...)

	target, err = d.GetTargetsByIndexes(labels...)
	if err != nil {
		err = fmt.Errorf("get targets: %w", err)
		return
	}

	return
}

func FromImages(w, h, d int, images ...[]float64) (outData *num.Data) {
	if len(images) < 1 {
		panic("images data is required")
	}

	if len(images[0]) != w*h*d {
		panic(fmt.Sprintf("image data length is not equal %d * %d * %d", w, h, d))
	}

	whd := w * h * d
	data := num.NewFloat64s(whd * len(images))

	offset := 0
	for _, image := range images {
		copy(data[offset:offset+whd], image)
		offset += whd
	}

	return num.NewWithValues(num.NewDims(w*h, d, len(images)), data)
}

func NewSeparateOneHotVectors(colsCount int) (vectors []*num.Data) {
	for i := 0; i < colsCount; i++ {
		data := num.NewFloat64s(colsCount)
		data[i] = 1.0
		vectors = append(vectors, num.NewWithValues(num.NewDims(colsCount), data))
	}
	return
}

func NewOneHotVectors(colsCount int, hots ...int) (outMatrix *num.Data) {
	rowsCount := len(hots)

	data := make([]float64, 0, colsCount*len(hots))

	for row := 0; row < rowsCount; row++ {
		vector := make([]float64, colsCount)
		vector[hots[row]] = 1

		data = append(data, vector...)
	}

	return num.NewWithValues(num.NewDims(colsCount, rowsCount), data)
}
