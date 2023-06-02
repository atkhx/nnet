package cifar_10

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"

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

var labels = []string{
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

func CreateTrainingDataset(datasetPath string) (*dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 training file: %w", err)
	}
	return result, nil
}

func CreateTestingDataset(datasetPath string) (*dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 testing file: %w", err)
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

		targets: NewSeparateOneHotVectors(len(labels)),
		labels:  labels,

		rgb: rgb,
	}
	return d, nil
}

type dataset struct {
	labels  []string
	targets []*num.Data
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

func (d *dataset) ReadSample(index int) (input, target *num.Data, err error) {
	label := d.labelsIdx[index]

	target, err = d.GetTarget(int(label))
	if err != nil {
		err = fmt.Errorf("get target %d failed: %w", label, err)
		return
	}

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

	return
}

func (d *dataset) ReadRandomSampleBatch(batchSize int) (input, target *num.Data, err error) {
	var images [][]float64
	var labels []int

	for i := 0; i < batchSize; i++ {
		index := rand.Intn(d.GetSamplesCount()) //nolint:gosec
		label := int(d.labelsIdx[index])

		var image []float64
		if d.rgb {
			image = d.images[index*ImageSizeRGB : (index+1)*ImageSizeRGB]
		} else {
			image = d.images[index*ImageSizeGray : (index+1)*ImageSizeGray]
		}

		images = append(images, image)
		labels = append(labels, label)
	}

	var chansCount = 1
	if d.rgb {
		chansCount = 3
	}

	input = FromImages(ImageWidth, ImageHeight, chansCount, images...)

	target, err = d.GetTargetsByIndexes(labels...)
	if err != nil {
		err = fmt.Errorf("get targets failed: %w", err)
		return
	}

	return
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
