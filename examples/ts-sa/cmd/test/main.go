package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/atkhx/nnet/examples/ts-sa/dataset"
	"github.com/atkhx/nnet/examples/ts-sa/pkg"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/ts-sa/config.json", "nn config file")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	batchSize := 1

	namesDataset := dataset.NewDataset(pkg.ContextLength, batchSize)
	namesDataset.ParseAlphabet(dataset.TinyShakespeare)

	seqModel := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		batchSize,
	)

	seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	inp, _ := namesDataset.ReadRandomSample()
	inputBytes := namesDataset.DecodeFloats(inp...)

	fmt.Println(strings.Repeat("-", 40))
	fmt.Println(string(namesDataset.Decode(inp.ToInt()...)))
	fmt.Println(strings.Repeat("-", 40))

	for j := 0; j < 10000; j++ {
		inputsFloat := namesDataset.EncodeToFloats(inputBytes...)

		out := seqModel.Forward(inputsFloat)
		output := out.Data[(pkg.ContextLength-1)*namesDataset.GetAlphabetSize():]
		output.Softmax()
		output.CumulativeSum()

		b := namesDataset.Decode(output.Multinomial())

		inputBytes = append(inputBytes, b...)
		inputBytes = inputBytes[len(b):]
		fmt.Print(string(b))
	}

	fmt.Println()
}
