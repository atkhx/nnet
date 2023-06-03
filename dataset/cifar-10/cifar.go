package cifar_10

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/atkhx/nnet/dataset"
	"github.com/atkhx/nnet/num"
)

var ErrorIndexOutOfRange = errors.New("index out of range")

const (
	ImageWidth  = 32
	ImageHeight = 32

	ImageSizeGray = ImageWidth * ImageHeight
	ImageSizeRGB  = ImageWidth * ImageHeight * 3
	SampleSize    = ImageSizeRGB + 1
)

var cifarLabels = []string{
	"plane",
	"auto",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
}

const (
	TrainImagesFileName = "cifar10-train-data.bin"
	TestImagesFileName  = "cifar10-test-data.bin"
)

func CreateTrainingDataset(datasetPath string) (*Dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 training file: %w", err)
	}
	return result, nil
}

func CreateTestingDataset(datasetPath string) (*Dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 testing file: %w", err)
	}
	return result, nil
}

func Open(filename string, rgb bool) (*Dataset, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	imagesCount := len(b) / SampleSize

	var images num.Float64s

	//nolint:gomnd
	if rgb {
		images = num.NewFloat64s(imagesCount * ImageSizeRGB)
		for i := 0; i < imagesCount; i++ {
			imageOffset := i * ImageSizeRGB
			sampleOffset := i * SampleSize
			for j := 0; j < ImageSizeRGB; j++ {
				images[imageOffset+j] = float64(b[sampleOffset+1+j]) / 255.0
			}
		}
	} else {
		images = num.NewFloat64s(imagesCount * ImageSizeGray)
		for i := 0; i < imagesCount; i++ {
			imageOffset := i * ImageSizeGray
			sampleOffset := i * SampleSize

			for j := 0; j < ImageSizeGray; j++ {
				R := float64(b[sampleOffset+1+j]) / 255.0
				G := float64(b[sampleOffset+1+j+ImageWidth]) / 255.0
				B := float64(b[sampleOffset+1+j+ImageWidth+ImageHeight]) / 255.0

				images[imageOffset+j] = (R + G + B) / 3
			}
		}
	}

	var labelsIdx = make([]byte, imagesCount)
	for i := 0; i < imagesCount; i++ {
		labelsIdx[i] = b[i*SampleSize]
	}

	d := &Dataset{
		labelsIdx:   labelsIdx,
		images:      images,
		imagesCount: imagesCount,

		targets: num.NewSeparateOneHotVectors(len(cifarLabels)),
		labels:  cifarLabels,

		rgb: rgb,
	}
	return d, nil
}

type Dataset struct {
	labels  []string
	targets []*num.Data
	images  []float64

	imagesCount int
	labelsIdx   []byte

	rgb bool
}

func (d *Dataset) GetSamplesCount() int {
	return d.imagesCount
}

func (d *Dataset) GetClasses() []string {
	return d.labels
}

func (d *Dataset) GetTarget(index int) (*num.Data, error) {
	if index > -1 && index < len(d.targets) {
		return d.targets[index], nil
	}
	return nil, ErrorIndexOutOfRange
}

func (d *Dataset) GetTargetsByIndexes(index ...int) (*num.Data, error) {
	return num.NewOneHotVectors(len(d.targets), index...), nil
}

func (d *Dataset) ReadSample(index int) (dataset.Sample, error) {
	label := d.labelsIdx[index]

	target, err := d.GetTarget(int(label))
	if err != nil {
		return dataset.Sample{}, fmt.Errorf("get target %d failed: %w", label, err)
	}

	var input *num.Data
	//nolint:gomnd
	if d.rgb {
		input = num.NewWithValues(
			num.NewDims(ImageWidth*ImageHeight, 3),
			d.images[index*ImageSizeRGB:(index+1)*ImageSizeRGB],
		)
	} else {
		input = num.NewWithValues(
			num.NewDims(ImageWidth*ImageHeight, 1),
			d.images[index*ImageSizeGray:(index+1)*ImageSizeGray],
		)
	}

	return dataset.Sample{
		Input:  input,
		Target: target,
	}, nil
}

func (d *Dataset) ReadRandomSampleBatch(batchSize int) (dataset.Sample, error) {
	images := make([]num.Float64s, 0, batchSize)
	labels := make([]int, 0, batchSize)

	for i := 0; i < batchSize; i++ {
		index := rand.Intn(d.GetSamplesCount()) //nolint:gosec

		if d.rgb {
			images = append(images, d.images[index*ImageSizeRGB:(index+1)*ImageSizeRGB])
		} else {
			images = append(images, d.images[index*ImageSizeGray:(index+1)*ImageSizeGray])
		}
		labels = append(labels, int(d.labelsIdx[index]))
	}

	var chansCount = 1
	if d.rgb {
		chansCount = 3
	}

	target, err := d.GetTargetsByIndexes(labels...)
	if err != nil {
		return dataset.Sample{}, fmt.Errorf("get targets failed: %w", err)
	}

	return dataset.Sample{
		Input:  num.FromImages(ImageWidth, ImageHeight, chansCount, images...),
		Target: target,
	}, nil
}
