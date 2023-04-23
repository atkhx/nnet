package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/atkhx/nnet/examples/namesgen-pos/dataset"
	"github.com/atkhx/nnet/examples/namesgen-pos/pkg"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/namesgen-pos/config.json", "nn config file")
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

	for j := 0; j < 100; j++ {
		inputBytes := bytes.Repeat([]byte{'.'}, namesDataset.GetContextSize())
		name := ""
		for k := 0; k < 50; k++ {
			inputsFloat := namesDataset.EncodeToFloats(inputBytes...)

			output := seqModel.Forward(inputsFloat).GetData()
			output.Softmax()
			output.CumulativeSum()

			b := namesDataset.Decode(output.Multinomial())

			inputBytes = append(inputBytes, b...)
			inputBytes = inputBytes[len(b):]

			if b[len(b)-1] == '.' {
				if name != "" {
					fmt.Println(name)
				}
				break
			}

			name += string(b)
		}
	}

	fmt.Println()
}
