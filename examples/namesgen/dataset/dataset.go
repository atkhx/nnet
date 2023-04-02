package dataset

import (
	"bytes"
	_ "embed"
	"math/rand"
	"sort"

	"github.com/atkhx/nnet/data"
)

//go:embed names.txt
var Names []byte

const (
	NamesContextSize   = 5
	NamesMiniBatchSize = 30
)

func NewDataset(contextSize, miniBatchSize int, bytes []byte) *Dataset {
	result := &Dataset{
		contextSize:   contextSize,
		miniBatchSize: miniBatchSize,
	}

	result.parseAlphabet(bytes)
	result.prepareDataset(bytes)

	return result
}

type Dataset struct {
	index map[byte]int
	chars []byte

	contextSize   int
	miniBatchSize int

	alphabetSize int
	samplesCount int

	inputs  []float64
	targets []float64
}

func (d *Dataset) GetSamplesCount() int {
	return d.samplesCount
}

func (d *Dataset) GetContextSize() int {
	return d.contextSize
}

func (d *Dataset) GetAlphabetSize() int {
	return d.alphabetSize
}

func (d *Dataset) GetMiniBatchSize() int {
	return d.miniBatchSize
}

func (d *Dataset) parseAlphabet(bytes []byte) {
	d.index = map[byte]int{}
	d.chars = []byte{}

	// Prepare index and collect unique set of chars
	for _, b := range bytes {
		if _, ok := d.index[b]; !ok {
			d.index[b] = 0
			d.chars = append(d.chars, b)
		}
	}

	// Sort unique set of chars (expect than "\n" will pop to the first place)
	sort.Slice(d.chars, func(i, j int) bool {
		return d.chars[i] < d.chars[j]
	})

	// Replace the first char value to special symbol "."
	d.chars[0] = '.'

	// Collect total alphabet size
	d.alphabetSize = len(d.chars)

	// Fill index map with actual chars positions
	for i, b := range d.chars {
		d.index[b] = i
	}
}

func (d *Dataset) prepareDataset(rawBytes []byte) {
	words := bytes.Split(rawBytes, []byte("\n"))
	input := bytes.Repeat([]byte{'.'}, d.contextSize)

	for _, word := range words {
		word := append(word, '.')
		for i := 0; i < len(word)-1; i++ {
			input = append(input, word[i])
			input = input[1:]

			d.inputs = append(d.inputs, data.NewOneHotVectors(d.alphabetSize, d.Encode(input...)...).Data.Data...)
			d.targets = append(d.targets, data.NewOneHotVectors(d.alphabetSize, d.Encode(word[i+1])...).Data.Data...)
		}
	}

	d.samplesCount = len(d.targets) / d.alphabetSize
}

func (d *Dataset) Encode(chars ...byte) []int {
	indexes := make([]int, len(chars))
	for i, v := range chars {
		indexes[i] = d.index[v]
	}
	return indexes
}

func (d *Dataset) Decode(pos ...int) []byte {
	result := make([]byte, len(pos))
	for i, p := range pos {
		result[i] = d.chars[p]
	}
	return result
}

func (d *Dataset) ReadRandomSampleBatch() (input, target *data.Data, err error) {
	pos := rand.Intn(d.GetSamplesCount() - d.miniBatchSize - 1)
	inputSampleSize := d.alphabetSize * d.contextSize

	sampleInputs := make([]float64, inputSampleSize*d.miniBatchSize)
	copy(sampleInputs, d.inputs[pos*inputSampleSize:(pos+d.miniBatchSize)*inputSampleSize])

	sampleTargets := make([]float64, d.alphabetSize*d.miniBatchSize)
	copy(sampleTargets, d.targets[pos*d.alphabetSize:(pos+d.miniBatchSize)*d.alphabetSize])

	input = data.WrapData(d.alphabetSize, d.contextSize*d.miniBatchSize, 1, sampleInputs)
	target = data.WrapData(d.alphabetSize, d.miniBatchSize, 1, sampleTargets)

	return
}
