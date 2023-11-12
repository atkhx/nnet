package dataset

import (
	"bufio"
	_ "embed"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"

	"github.com/atkhx/nnet/num/dev/native"
)

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

var sourceTxt = "./gpt/data/ruwiki12.txt"
var sourceAlphabet = "./gpt/data/ruwiki12.alphabet"

//var sourceTxt = "./gpt/data/rus_sentences.txt"
//var sourceAlphabet = "./gpt/data/rus_sentences.alphabet"

func (d *Dataset) ParseAlphabet() {
	f2, err := os.OpenFile(sourceAlphabet, os.O_RDONLY, os.ModePerm)
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

	d.alphabetSize = len(d.tokens)
}

func (d *Dataset) ParseTokens() {
	f, err := os.OpenFile(sourceTxt, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	b, err := io.ReadAll(f)

	//b := make([]byte, 60_000_000)
	//n, err := f.ReadAt(b, 0)
	//n, err := f.ReadAt(b, 30_000_000)
	//n, err := f.ReadAt(b, 60_000_000)
	//n, err := f.ReadAt(b, 90_000_000)
	//n, err := f.ReadAt(b, 120_000_000)
	if err != nil {
		log.Fatalln(err)
	}
	//b = b[:n]
	fmt.Println("length b", len(b))

	rawBytes := []rune(string(b))
	d.rawRunes = rawBytes
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

func (d *Dataset) EncodeToFloats(chars ...Token) []float32 {
	indexes := make([]float32, len(chars))
	for i, v := range chars {
		indexes[i] = float32(d.tokenCodes[v])
	}
	return indexes
}

type Tokens []Token

func (t Tokens) Copy() Tokens {
	dst := make(Tokens, len(t))
	copy(dst, t)
	return dst
}

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

func (d *Dataset) DecodeFloats(pos ...float32) []Token {
	result := make([]Token, len(pos))
	for i, p := range pos {
		result[i] = d.tokens[int(p)]
	}
	return result
}

func (d *Dataset) ReadRandomSampleBatch() (sampleInputs, sampleTargets native.Float32s) {
	sampleInputs = make(native.Float32s, d.contextSize*d.miniBatchSize)
	sampleTargets = make(native.Float32s, d.contextSize*d.miniBatchSize)

	tokens := make([]Token, d.contextSize)
	tokenLength := 1

	for b := 0; b < d.miniBatchSize; b++ {
		pos := rand.Intn(len(d.rawRunes) - (d.contextSize * tokenLength) - 1)

		chunk := d.rawRunes[pos : (pos+1)+(d.contextSize*tokenLength)]
		for i := 0; i < d.contextSize; i++ {
			tokens[i] = Token(chunk[i*tokenLength : (i+1)*tokenLength])
		}

		sampleInputs := sampleInputs[b*d.contextSize : (b+1)*d.contextSize]
		copy(sampleInputs, d.EncodeToFloats(tokens...))

		for i := 1; i < d.contextSize+1; i++ {
			tokens[i-1] = Token(chunk[i*tokenLength : (i+1)*tokenLength])
		}

		sampleTargets := sampleTargets[b*d.contextSize : (b+1)*d.contextSize]
		copy(sampleTargets, d.EncodeToFloats(tokens...))
	}

	return sampleInputs, sampleTargets
}
