package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/atkhx/nnet/examples/transformer/dataset"
	"github.com/atkhx/nnet/examples/transformer/pkg"
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

	inputIndexes := trainDataset.EncodeString(` `)
	inputTokens := trainDataset.Decode(inputIndexes...)

	fmt.Print(trainDataset.Decode(inputIndexes...))

	for j := 0; j < 10000; j++ {
		inputsFloat := trainDataset.EncodeToFloats(inputTokens...)
		copy(model.GetInput().Data, inputsFloat)

		pipeline.Forward()

		pos := len(inputTokens) - 1
		if pos < 0 {
			pos = 0
		}

		logits := output.Data[pos*trainDataset.GetAlphabetSize() : (pos+1)*trainDataset.GetAlphabetSize()]
		logits.Softmax() // probs
		logits.CumulativeSum()
		f := logits.Multinomial()
		//f := sampleWithTemperature(logits, 0.9)
		b := trainDataset.Decode(f)

		inputTokens = append(inputTokens, b...)
		if len(inputTokens) > pkg.ContextLength {
			l := len(inputTokens)
			inputTokens = inputTokens[l-pkg.ContextLength:]
		}

		fmt.Print(b)
	}

	fmt.Println()
}

func sampleWithTemperature(logits num.Float64s, temperature float64) int {
	for i := 0; i < len(logits); i++ {
		logits[i] /= temperature
	}

	return sampleFromDistribution(softmax(logits))
}

func softmax(logits []float64) []float64 {
	probs := make([]float64, len(logits))
	sum := 0.0

	for _, logit := range logits {
		sum += math.Exp(logit)
	}

	for i, logit := range logits {
		probs[i] = math.Exp(logit) / sum
	}
	return probs
}

func sampleFromDistribution(probs []float64) int {
	r := rand.Float64()
	cumulativeProb := 0.0

	for i, prob := range probs {
		cumulativeProb += prob
		if r <= cumulativeProb {
			return i
		}
	}

	panic("unreachable code")
}
