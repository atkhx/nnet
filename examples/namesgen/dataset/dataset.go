package dataset

import (
	"bytes"
	_ "embed"
	"math/rand"
	"sort"

	"github.com/atkhx/nnet/num"
)

//go:embed names.txt
var Names []byte

const (
	NamesContextSize   = 5
	NamesMiniBatchSize = 15
)

func NewDataset(contextSize, miniBatchSize int) *Dataset {
	result := &Dataset{
		contextSize:   contextSize,
		miniBatchSize: miniBatchSize,
	}

	return result
}

type Dataset struct {
	index map[byte]int
	chars []byte

	contextSize   int
	miniBatchSize int

	alphabetSize int
	samplesCount int

	inputs  num.Float64s
	targets num.Float64s
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

func (d *Dataset) ParseAlphabet(bytes []byte) {
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

func (d *Dataset) PrepareDataset(rawBytes []byte) {
	words := bytes.Split(rawBytes, []byte("\n"))
	input := bytes.Repeat([]byte{'.'}, d.contextSize)

	for _, word := range words {
		word := append(word, '.')
		for i := 0; i < len(word)-1; i++ {
			input = append(input, word[i])
			input = input[1:]

			d.inputs = append(d.inputs, num.NewOneHotVectors(d.alphabetSize, d.Encode(input...)...)...)
			d.targets = append(d.targets, num.NewOneHotVectors(d.alphabetSize, d.Encode(word[i+1])...)...)
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

func (d *Dataset) ReadRandomSampleBatch() (num.Float64s, num.Float64s) {
	pos := rand.Intn(d.GetSamplesCount() - d.miniBatchSize - 1)
	inputSampleSize := d.alphabetSize * d.contextSize

	sampleInputs := make(num.Float64s, inputSampleSize*d.miniBatchSize)
	copy(sampleInputs, d.inputs[pos*inputSampleSize:(pos+d.miniBatchSize)*inputSampleSize])

	sampleTargets := make(num.Float64s, d.alphabetSize*d.miniBatchSize)
	copy(sampleTargets, d.targets[pos*d.alphabetSize:(pos+d.miniBatchSize)*d.alphabetSize])

	return sampleInputs, sampleTargets
}
