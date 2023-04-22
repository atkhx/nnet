package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/atkhx/nnet/examples/ts/dataset"
	"github.com/atkhx/nnet/examples/ts/pkg"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/ts/config.json", "nn config file")
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
	namesDataset.ParseAlphabet(dataset.TinyShakespeare)

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
	inp, _ := namesDataset.ReadRandomSample()
	inputBytes := namesDataset.DecodeFloats(inp...)

	fmt.Println(strings.Repeat("-", 40))
	fmt.Println(string(namesDataset.Decode(inp.ToInt()...)))
	fmt.Println(strings.Repeat("-", 40))
	//fmt.Print(string(namesDataset.Decode(inp.ToInt()...)))

	for j := 0; j < 10000; j++ {
		inputsFloat := namesDataset.EncodeToFloats(inputBytes...)

		seqModel.Forward(inputsFloat, output)
		output.Softmax()
		output.CumulativeSum()

		b := namesDataset.Decode(output.Multinomial())

		inputBytes = append(inputBytes, b...)
		inputBytes = inputBytes[len(b):]
		fmt.Print(string(b))
	}

	fmt.Println()
}
