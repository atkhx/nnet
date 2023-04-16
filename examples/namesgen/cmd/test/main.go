package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/atkhx/nnet/examples/namesgen/dataset"
	"github.com/atkhx/nnet/examples/namesgen/pkg"
	"github.com/atkhx/nnet/num"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/namesgen/config.json", "nn config file")
	flag.Parse()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	batchSize := 1

	namesDataset := dataset.NewDataset(dataset.NamesContextSize, batchSize)
	namesDataset.ParseAlphabet(dataset.Names)

	seqModel := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		namesDataset.GetContextSize(),
		batchSize,
	)

	seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	output := seqModel.NewOutput()

	for j := 0; j < 100; j++ {
		inputBytes := bytes.Repeat([]byte{'.'}, namesDataset.GetContextSize())
		for k := 0; k < 50; k++ {
			inputs := num.NewOneHotVectors(namesDataset.GetAlphabetSize(), namesDataset.Encode(inputBytes...)...)
			seqModel.Forward(inputs, output)
			output.Softmax()

			b := namesDataset.Decode(output.Multinomial())

			inputBytes = append(inputBytes, b...)
			inputBytes = inputBytes[len(b):]

			if b[len(b)-1] == '.' {
				fmt.Println()
				break
			}
			fmt.Print(string(b))
		}
	}

	fmt.Println()
}
