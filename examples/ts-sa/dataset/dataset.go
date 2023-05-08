package dataset

import (
	_ "embed"
	"math/rand"
	"sort"

	"github.com/atkhx/nnet/num"
)

//go:embed tinyshakespeare.txt
var TinyShakespeare []byte

const (
	NamesContextSize   = 32
	NamesMiniBatchSize = 10
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

	rawBytes []byte

	//inputs  num.Float64s
	//targets num.Float64s
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
	d.rawBytes = bytes

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

	// Collect total alphabet size
	d.alphabetSize = len(d.chars)

	// Fill index map with actual chars positions
	for i, b := range d.chars {
		d.index[b] = i
	}
}

func (d *Dataset) Encode(chars ...byte) []int {
	indexes := make([]int, len(chars))
	for i, v := range chars {
		indexes[i] = d.index[v]
	}
	return indexes
}

func (d *Dataset) EncodeToFloats(chars ...byte) []float64 {
	indexes := make([]float64, len(chars))
	for i, v := range chars {
		indexes[i] = float64(d.index[v])
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

func (d *Dataset) DecodeFloats(pos ...float64) []byte {
	result := make([]byte, len(pos))
	for i, p := range pos {
		result[i] = d.chars[int(p)]
	}
	return result
}

func (d *Dataset) ReadRandomSampleBatch() (sampleInputs, sampleTargets num.Float64s) {
	inputSampleSize := d.contextSize
	sampleInputs = make(num.Float64s, inputSampleSize*d.miniBatchSize)
	sampleTargets = make(num.Float64s, d.alphabetSize*d.miniBatchSize)

	for b := 0; b < d.miniBatchSize; b++ {
		pos := rand.Intn(len(d.rawBytes) - inputSampleSize - 1)

		sampleInputs := sampleInputs[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleInputs, d.EncodeToFloats(d.rawBytes[pos:pos+inputSampleSize]...))

		target := num.NewOneHotVectors(d.alphabetSize, d.Encode(d.rawBytes[pos+inputSampleSize])...)
		sampleTarget := sampleTargets[b*d.alphabetSize : (b+1)*d.alphabetSize]
		copy(sampleTarget, target)
	}

	return sampleInputs, sampleTargets
}

func (d *Dataset) ReadRandomSample() (sampleInputs, sampleTargets num.Float64s) {
	inputSampleSize := d.contextSize
	sampleInputs = make(num.Float64s, inputSampleSize)
	sampleTargets = make(num.Float64s, d.alphabetSize)

	for b := 0; b < 1; b++ {
		pos := rand.Intn(len(d.rawBytes) - inputSampleSize - 1)

		sampleInputs := sampleInputs[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleInputs, d.EncodeToFloats(d.rawBytes[pos:pos+inputSampleSize]...))

		target := num.NewOneHotVectors(d.alphabetSize, d.Encode(d.rawBytes[pos+inputSampleSize])...)
		sampleTarget := sampleTargets[b*d.alphabetSize : (b+1)*d.alphabetSize]
		copy(sampleTarget, target)
	}

	return sampleInputs, sampleTargets
}
