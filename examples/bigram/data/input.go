package data

import (
	"bytes"
	_ "embed"
	"sort"

	"github.com/atkhx/nnet/data"
)

//go:embed names.txt
var Bytes []byte

var Chars []byte
var Index map[byte]int

var AlphabetSize int
var WordLen = 5

var Samples []Sample

type Sample struct {
	Input  *data.Data
	Target *data.Data
}

func init() {
	Index = map[byte]int{}

	for _, b := range Bytes {
		if _, ok := Index[b]; !ok {
			Index[b] = 0
			Chars = append(Chars, b)
		}
	}

	sort.Slice(Chars, func(i, j int) bool {
		return Chars[i] < Chars[j]
	})

	AlphabetSize = len(Chars)

	Chars[0] = '.'

	for i, b := range Chars {
		Index[b] = i
	}

	names := bytes.Split(Bytes, []byte("\n"))

	for _, name := range names {
		name := append(name, '.')
		inputBytes := bytes.Repeat([]byte{'.'}, WordLen)
		targetByte := byte(0)
		for i := 0; i < len(name)-1; i++ {
			inputBytes = append(inputBytes, name[i])
			inputBytes = inputBytes[1:]
			targetByte = name[i+1]

			Samples = append(Samples, Sample{
				Input:  data.MustCompileMultiOneHotsMatrix(AlphabetSize, Encode(inputBytes)...),
				Target: data.MustCompileOneHotVector(EncodeOne(targetByte), AlphabetSize),
			})

			//fmt.Println("inputBytes", string(inputBytes), "===>", string(targetByte))
			//fmt.Println(Encode(inputBytes))
			//return
		}

		//fmt.Println(string(name))
		//if nameIndex > 5 {
		//	break
		//}
	}
}

func Encode(chars []byte) []int {
	indexes := make([]int, len(chars))
	for i, v := range chars {
		indexes[i] = Index[v]
	}
	return indexes
}

func EncodeOne(char byte) int {
	return Index[char]
}

func EncodeToFloats(chars []byte) []float64 {
	indexes := make([]float64, len(chars))
	for i, v := range chars {
		indexes[i] = float64(Index[v])
	}
	return indexes
}
