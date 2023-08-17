package dataset

import (
	_ "embed"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/atkhx/nnet/num"

	//"github.com/sugarme/tokenizer"
	"github.com/cohere-ai/tokenizer"
)

func NewDataset(tk *tokenizer.Encoder, contextSize, miniBatchSize int) *Dataset {
	result := &Dataset{
		tk:            tk,
		contextSize:   contextSize,
		miniBatchSize: miniBatchSize,
	}

	return result
}

type Token string

type Dataset struct {
	//tk *tokenizer.Tokenizer
	tk *tokenizer.Encoder

	rawIndexes []int64

	contextSize   int
	miniBatchSize int
}

func (d *Dataset) OpenTrainingFile(filename string) error {
	f, err := os.OpenFile(filename, os.O_RDONLY, os.ModePerm)
	if err != nil {
		return err
	}
	defer f.Close()

	b := make([]byte, 50_000_000)
	if n, err := f.ReadAt(b, 0); err != nil {
		return err
	} else {
		b = b[:n]
	}

	//b, err := io.ReadAll(f)
	//if err != nil {
	//	return err
	//}

	t := time.Now()
	i, _ := d.tk.Encode(string(b))
	fmt.Println("time to tokenize:", time.Since(t))
	d.rawIndexes = i

	//t := time.Now()
	//i, err := d.tk.EncodeSingle(string(b), true)
	//if err != nil {
	//	return err
	//}
	//fmt.Println("time to tokenize:", time.Since(t))
	//
	//d.rawIndexes = i.Ids

	fmt.Println("data size", len(d.rawIndexes), "indexes")
	return nil
}

func (d *Dataset) GetContextSize() int {
	return d.contextSize
}

func (d *Dataset) GetAlphabetSize() int {
	return int(d.tk.VocabSize)
	//return d.tk.GetVocabSize(false)
}

func (d *Dataset) GetMiniBatchSize() int {
	return d.miniBatchSize
}

func (d *Dataset) EncodeString(value string) []int64 {
	en, _ := d.tk.Encode(value)
	return en
	//en, err := d.tk.EncodeSingle(value, true)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//return en.Ids
}

func (d *Dataset) EncodeToFloats(value string) []float64 {
	idx, _ := d.tk.Encode(value)
	//idx := d.EncodeString(value)
	res := make([]float64, len(idx))
	for i, index := range idx {
		res[i] = float64(index)
	}
	return res
}

func (d *Dataset) Decode(idx ...int64) string {
	return d.tk.Decode(idx)
	//return d.tk.Decode(idx, false)
}

func (d *Dataset) DecodeFloats(pos ...float64) string {
	idx := make([]int64, len(pos))
	for i, f := range pos {
		idx[i] = int64(f)
	}
	return d.Decode(idx...)
}

func (d *Dataset) toFloats(ints []int64) []float64 {
	res := make([]float64, len(ints))
	for i, v := range ints {
		res[i] = float64(v)
	}
	return res
}

func (d *Dataset) ReadRandomSampleBatch() (sampleInputs, sampleTargets num.Float64s) {
	inputSampleSize := d.contextSize

	sampleInputs = make(num.Float64s, inputSampleSize*d.miniBatchSize)
	sampleTargets = make(num.Float64s, inputSampleSize*d.miniBatchSize)

	tokenLength := 1

	for b := 0; b < d.miniBatchSize; b++ {
		pos := rand.Intn(len(d.rawIndexes) - (d.contextSize * tokenLength) - 1)

		chunk := d.rawIndexes[pos : (pos+1)+(inputSampleSize*tokenLength)]

		sampleChunk := chunk[:len(chunk)-1]
		targetChunk := chunk[1:len(chunk)]

		sampleInputs := sampleInputs[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleInputs, d.toFloats(sampleChunk))

		//fmt.Println("data")
		//fmt.Println(d.Decode(sampleChunk...))
		//fmt.Println(strings.Repeat("-", 40))

		sampleTargets := sampleTargets[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleTargets, d.toFloats(targetChunk))
		//fmt.Println(d.Decode(targetChunk...))
		//fmt.Println(strings.Repeat("-", 40))
	}

	return sampleInputs, sampleTargets
}
