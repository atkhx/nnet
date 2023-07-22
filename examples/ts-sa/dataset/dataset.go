package dataset

import (
	"bufio"
	_ "embed"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"

	"github.com/atkhx/nnet/num"
)

// дддлgo:embed spring.txt
//
// ddgo:embed tinyshakespeare.txt
// var TinyShakespeare []byte
//
//fffgo:embed ruwiki1.txt
var RuWiki1 []rune

func NewDataset(contextSize, miniBatchSize int) *Dataset {
	result := &Dataset{
		contextSize:   contextSize,
		miniBatchSize: miniBatchSize,
	}

	return result
}

type Token string

type Dataset struct {
	tokenCodes map[Token]int
	tokens     []Token
	rawRunes   []rune

	contextSize   int
	miniBatchSize int

	alphabetSize int
	samplesCount int
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

var wiki = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki1.txt"
var wikiAlphabet = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki1.alphabet"

func (d *Dataset) ParseAlphabet(_ []rune) {
	f2, err := os.OpenFile(wikiAlphabet, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer f2.Close()

	d.tokenCodes = map[Token]int{}
	d.tokens = []Token{}

	reader := bufio.NewReader(f2)

	var p = 0
	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatalln(err)
		}

		d.tokens = append(d.tokens, Token(r))
		d.tokenCodes[Token(r)] = p
		p++
	}

	f, err := os.OpenFile(wiki, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	b := make([]byte, 1000_000)
	//n, err := f.ReadAt(b, 6000_000)
	n, err := f.ReadAt(b, 5000_000)
	if err != nil {
		log.Fatalln(err)
	}
	b = b[:n]
	fmt.Println("length b", len(b))

	rawBytes := []rune(string(b))
	d.rawRunes = rawBytes

	d.alphabetSize = len(d.tokens)
	fmt.Println("d.alphabetSize", d.alphabetSize)
}

func (d *Dataset) EncodeString(value string) []int {
	indexes := make([]int, len([]rune(value)))
	for i, v := range []rune(value) {
		indexes[i] = d.tokenCodes[Token(v)]
	}
	return indexes
}

func (d *Dataset) Encode(chars ...Token) []int {
	indexes := make([]int, len(chars))
	for i, v := range chars {
		indexes[i] = d.tokenCodes[v]
	}
	return indexes
}

func (d *Dataset) EncodeToFloats(chars ...Token) []float64 {
	indexes := make([]float64, len(chars))
	for i, v := range chars {
		indexes[i] = float64(d.tokenCodes[v])
	}
	return indexes
}

type Tokens []Token

func (t Tokens) String() string {
	result := ""
	for _, token := range t {
		result += string(token)
	}
	return result
}

func (d *Dataset) Decode(pos ...int) Tokens {
	result := make([]Token, len(pos))
	for i, p := range pos {
		t := d.tokens[p]
		result[i] = t
	}
	return result
}

func (d *Dataset) DecodeFloats(pos ...float64) []Token {
	result := make([]Token, len(pos))
	for i, p := range pos {
		result[i] = d.tokens[int(p)]
	}
	return result
}

func (d *Dataset) ReadRandomSampleBatch() (sampleInputs, sampleTargets num.Float64s) {
	inputSampleSize := d.contextSize

	sampleInputs = make(num.Float64s, inputSampleSize*d.miniBatchSize)
	sampleTargets = make(num.Float64s, inputSampleSize*d.miniBatchSize)

	tokens := make([]Token, d.contextSize)
	tokenLength := 1

	for b := 0; b < d.miniBatchSize; b++ {
		pos := rand.Intn(len(d.rawRunes) - (d.contextSize * tokenLength) - 1)

		chunk := d.rawRunes[pos : (pos+1)+(inputSampleSize*tokenLength)]
		for i := 0; i < d.contextSize; i++ {
			tokens[i] = Token(chunk[i*tokenLength : (i+1)*tokenLength])
		}

		sampleInputs := sampleInputs[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleInputs, d.EncodeToFloats(tokens...))

		for i := 1; i < d.contextSize+1; i++ {
			tokens[i-1] = Token(chunk[i*tokenLength : (i+1)*tokenLength])
		}

		sampleTargets := sampleTargets[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleTargets, d.EncodeToFloats(tokens...))
	}

	return sampleInputs, sampleTargets
}

func (d *Dataset) ReadRandomSampleBatch2() (sampleInputs, sampleTargets num.Float64s) {
	inputSampleSize := d.contextSize

	sampleInputs = make(num.Float64s, inputSampleSize*d.miniBatchSize)
	sampleTargets = make(num.Float64s, inputSampleSize*d.miniBatchSize)

	tokens := make([]Token, d.contextSize)
	tokenLength := 1

	batchPos := rand.Intn(len(d.rawRunes) - (d.miniBatchSize * d.contextSize * tokenLength) - 1)

	for b := 0; b < d.miniBatchSize; b++ {
		pos := batchPos + b*d.contextSize*tokenLength
		chunk := d.rawRunes[pos : (pos+1)+(inputSampleSize*tokenLength)]
		for i := 0; i < d.contextSize; i++ {
			tokens[i] = Token(chunk[i*tokenLength : (i+1)*tokenLength])
		}

		sampleInputs := sampleInputs[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleInputs, d.EncodeToFloats(tokens...))

		for i := 1; i < d.contextSize+1; i++ {
			tokens[i-1] = Token(chunk[i*tokenLength : (i+1)*tokenLength])
		}

		sampleTargets := sampleTargets[b*inputSampleSize : (b+1)*inputSampleSize]
		copy(sampleTargets, d.EncodeToFloats(tokens...))
	}

	return sampleInputs, sampleTargets
}
