package dataset

import (
	"bytes"
	_ "embed"
	"fmt"
	"math/rand"
	"os"
)

func NewDataset(
	contextSize,
	miniBatchSize int,
	sourceTxt string,
	sourceAlphabet string,
) *Dataset {
	return &Dataset{
		contextSize:    contextSize,
		miniBatchSize:  miniBatchSize,
		sourceTxt:      sourceTxt,
		sourceAlphabet: sourceAlphabet,
	}
}

type Dataset struct {
	tokenCodes map[rune]int
	tokens     []rune

	rawFloats []float32

	contextSize   int
	miniBatchSize int
	alphabetSize  int

	sourceAlphabet string
	sourceTxt      string
}

func (d *Dataset) GetAlphabetSize() int {
	return d.alphabetSize
}

func (d *Dataset) ParseAlphabet() (e error) {
	b, err := os.ReadFile(d.sourceAlphabet)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	d.tokenCodes = map[rune]int{}
	d.tokens = []rune{}

	for _, r := range bytes.Runes(b) {
		d.tokenCodes[r] = len(d.tokens)
		d.tokens = append(d.tokens, r)
	}

	d.alphabetSize = len(d.tokens)
	return nil
}

func (d *Dataset) ParseTokens() (e error) {
	b, err := os.ReadFile(d.sourceTxt)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}
	d.rawFloats = d.EncodeToFloats(bytes.Runes(b)...)
	return nil
}

func (d *Dataset) EncodeString(value string) []int {
	indexes := make([]int, len([]rune(value)))
	for i, v := range []rune(value) {
		indexes[i] = d.tokenCodes[v]
	}
	return indexes
}

func (d *Dataset) Encode(chars ...rune) []int {
	indexes := make([]int, len(chars))
	for i, v := range chars {
		indexes[i] = d.tokenCodes[v]
	}
	return indexes
}

func (d *Dataset) EncodeToFloats(chars ...rune) []float32 {
	indexes := make([]float32, len(chars))
	for i, v := range chars {
		indexes[i] = float32(d.tokenCodes[v])
	}
	return indexes
}

func (d *Dataset) Decode(pos ...int) []rune {
	result := make([]rune, len(pos))
	for i, p := range pos {
		t := d.tokens[p]
		result[i] = t
	}
	return result
}

func (d *Dataset) DecodeFloats(pos ...float32) []rune {
	result := make([]rune, len(pos))
	for i, p := range pos {
		result[i] = d.tokens[int(p)]
	}
	return result
}

func (d *Dataset) ReadRandomSampleBatch() (sampleInputs, sampleTargets []float32) {
	lln := d.miniBatchSize * d.contextSize
	pos := rand.Intn(len(d.rawFloats) - lln - 1)

	return d.rawFloats[pos : pos+lln], d.rawFloats[pos+1 : pos+1+lln]
}
