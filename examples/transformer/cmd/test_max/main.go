package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/atkhx/nnet/examples/transformer/dataset"
	"github.com/atkhx/nnet/examples/transformer/pkg"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/transformer/config.json", "nn config file")
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

	trainDataset := dataset.NewDataset(pkg.ContextLength, batchSize)
	trainDataset.ParseAlphabet()

	model := pkg.CreateNN(
		trainDataset.GetAlphabetSize(),
		batchSize,
		nil,
	)

	output := model.Compile()
	if err := model.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := num.NewPipeline(output)

	inputIndexes := trainDataset.EncodeString(`Россия`)
	inputTokens := trainDataset.Decode(inputIndexes...)

	fmt.Print(trainDataset.Decode(inputIndexes...))

	getNextProbs := getNextProbsFunc(
		trainDataset,
		model,
		pipeline,
		output,
	)

	for j := 0; j < 10000; j++ {
		probs, tokens := getNextProbs(inputTokens)

		f := probs.MaxIndex()
		b := tokens[f]

		inputTokens = append(inputTokens, b)
		if len(inputTokens) > pkg.ContextLength {
			l := len(inputTokens)
			inputTokens = inputTokens[l-pkg.ContextLength:]
		}

		fmt.Print(b)
	}
}

func getNextProbsFunc(
	trainDataset *dataset.Dataset,
	model *model.Sequential,
	pipeline *num.Pipeline,
	output *num.Data,
) func(inputBytes dataset.Tokens) (num.Float64s, []dataset.Token) {
	return func(inputBytes dataset.Tokens) (num.Float64s, []dataset.Token) {
		inputsFloat := trainDataset.EncodeToFloats(inputBytes...)

		copy(model.GetInput().Data, inputsFloat)
		pipeline.Forward()

		pos := len(inputBytes) - 1
		if pos < 0 {
			pos = 0
		}

		outputData := output.Data[pos*trainDataset.GetAlphabetSize() : (pos+1)*trainDataset.GetAlphabetSize()]
		outputData.Softmax()

		probs := outputData.Copy()
		tokens := []dataset.Token{}

		for i := range probs {
			tokens = append(tokens, trainDataset.Decode(i)[0])
		}

		return probs, tokens
	}
}
