package cifar_10

import (
	"fmt"
	"os"
	"strings"

	"github.com/pkg/errors"

	"github.com/atkhx/nnet/data"
)

var ErrorIndexOutOfRange = errors.New("index out of range")

const (
	ImageWidth  = 32
	ImageHeight = 32

	ImageSizeGray = ImageWidth * ImageHeight
	ImageSizeRGB  = ImageWidth * ImageHeight * 3
	SampleSize    = ImageSizeRGB + 1
)

var labels = []string{
	"airplane",
	"automobile",
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

func CreateTrainingDataset(datasetPath string) (*dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, errors.Wrap(err, "can't open cifar-10 training file")
	}
	return result, nil
}

func CreateTestingDataset(datasetPath string) (*dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, errors.Wrap(err, "can't open cifar-10 testing file")
	}
	return result, nil
}

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
				images[imageOffset+j] = float64(b[sampleOffset+1+j]) / 255.0
			}
		}
	} else {
		images = make([]float64, imagesCount*ImageSizeGray)
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

	d := &dataset{
		labelsIdx:   labelsIdx,
		images:      images,
		imagesCount: imagesCount,

		targets: data.MustCompileOneHotVectors(len(labels)),
		labels:  labels,

		rgb: rgb,
	}
	return d, nil
}

type dataset struct {
	labels  []string
	targets []*data.Data
	images  []float64

	imagesCount int
	labelsIdx   []byte

	rgb bool
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
