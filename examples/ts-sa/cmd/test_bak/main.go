package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/atkhx/nnet/examples/ts-sa/dataset"
	"github.com/atkhx/nnet/examples/ts-sa/pkg"
	"github.com/atkhx/nnet/num"
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
	namesDataset.ParseAlphabet()
	//namesDataset.ParseTokens()

	seqModel := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		batchSize,
	)

	output := seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := num.NewPipeline(output)

	inp := namesDataset.EncodeString("")
	inputBytes := namesDataset.Decode(inp...)

	fmt.Print(namesDataset.Decode(inp...))

	for j := 0; j < 10000; j++ {
		inputsFloat := namesDataset.EncodeToFloats(inputBytes...)

		copy(seqModel.GetInput().Data, inputsFloat)
		pipeline.Forward()

		pos := len(inputBytes) - 1
		if pos < 0 {
			pos = 0
		}

		output := output.Data[pos*namesDataset.GetAlphabetSize() : (pos+1)*namesDataset.GetAlphabetSize()]
		output.Softmax()
		output.CumulativeSum()
		b := namesDataset.Decode(output.Multinomial())

		inputBytes = append(inputBytes, b...)
		if len(inputBytes) > pkg.ContextLength {
			l := len(inputBytes)
			inputBytes = inputBytes[l-pkg.ContextLength:]
		}

		fmt.Print(b)
	}

	fmt.Println()
}
