package mnist

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/num"
)

const (
	ImageWidth  = 28
	ImageHeight = 28
	ImageDepth  = 1
	ImageSize   = ImageWidth * ImageHeight * ImageDepth

	TrainImagesFileName = "train-images-idx3-ubyte"
	TrainLabelsFileName = "train-labels-idx1-ubyte"

	TestImagesFileName = "t10k-images-idx3-ubyte"
	TestLabelsFileName = "t10k-labels-idx1-ubyte"

	imagesFileOffset = 4 * 4
	labelsFileOffset = 2 * 4
)

var mnistLabels = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

var ErrOutOfRange = errors.New("index out of range")

func CreateTrainingDataset(datasetPath string) (*Dataset, error) {
	return New(
		fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName),
		fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainLabelsFileName),
	)
}

func CreateTestingDataset(datasetPath string) (*Dataset, error) {
	return New(
		fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestImagesFileName),
		fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestLabelsFileName),
	)
}

func New(imagesFileName, labelsFileName string) (*Dataset, error) {
	images, imagesCount, err := readImages(imagesFileName)
	if err != nil {
		return nil, fmt.Errorf("read images: %w", err)
	}

	labels, labelsCount, err := readLabels(labelsFileName)
	if err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}

	if imagesCount != labelsCount {
		return nil, fmt.Errorf("images count (%d) not equals labels count (%d)", imagesCount, labelsCount)
	}

	return &Dataset{
		images:       images,
		labels:       labels,
		samplesCount: imagesCount,
	}, nil
}

type Dataset struct {
	images num.Float64s
	labels []byte

	samplesCount int
}

func (d *Dataset) GetSamplesCount() int {
	return d.samplesCount
}

func (d *Dataset) GetClasses() []string {
	return mnistLabels
}

func (d *Dataset) ReadSample(index int) (dataset.Sample, error) {
	if index < 0 || index > d.samplesCount {
		return dataset.Sample{}, fmt.Errorf("%w: index %d, count: %d", ErrOutOfRange, index, d.samplesCount)
	}

	image := d.images[index*ImageSize : (index+1)*ImageSize]
	label := int(d.labels[index])

	return dataset.Sample{
		Input:  num.FromImages(ImageWidth, ImageHeight, ImageDepth, image),
		Target: num.NewOneHotVectors(len(mnistLabels), label),
	}, nil
}

func (d *Dataset) ReadRandomSampleBatch(batchSize int) (dataset.Sample, error) {
	images := make([]num.Float64s, 0, batchSize)
	labels := make([]int, 0, batchSize)

	for i := 0; i < batchSize; i++ {
		index := rand.Intn(d.samplesCount) //nolint:gosec
		images = append(images, d.images[index*ImageSize:(index+1)*ImageSize])
		labels = append(labels, int(d.labels[index]))
	}

	return dataset.Sample{
		Input:  num.FromImages(ImageWidth, ImageHeight, ImageDepth, images...),
		Target: num.NewOneHotVectors(len(mnistLabels), labels...),
	}, nil
}

func readImages(filename string) (num.Float64s, int, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, 0, fmt.Errorf("read file: %w", err)
	}

	b = b[imagesFileOffset:]
	d := num.NewFloat64s(len(b))
	for i := 0; i < len(d); i++ {
		//nolint:gomnd
		d[i] = float64(b[i]) / 255.0
	}

	return d, (len(b)) / ImageSize, nil
}

func readLabels(filename string) ([]byte, int, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, 0, fmt.Errorf("read labels: %w", err)
	}

	b = b[labelsFileOffset:]
	return b, len(b), nil
}
