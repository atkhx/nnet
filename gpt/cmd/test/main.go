package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/atkhx/metal/nn/proc"
	"github.com/atkhx/nnet/gpt/dataset"
	"github.com/atkhx/nnet/gpt/pkg"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./gpt/config.json", "nn config file")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	batchSize := 1

	trainDataset := dataset.NewDataset(pkg.ContextLength, batchSize)
	trainDataset.ParseAlphabet()

	model := pkg.CreateInferenceModel(
		trainDataset.GetAlphabetSize(),
		batchSize,
		device,
	)

	output := model.Compile()
	inputs := model.GetInput()

	if err := model.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := device.GetInferencePipeline(output)

	inputIndexes := trainDataset.EncodeString(`Китай`)
	inputTokens := trainDataset.Decode(inputIndexes...)

	fmt.Print(trainDataset.Decode(inputIndexes...))
	for j := 0; j < pkg.ContextLength*10; j++ {
		inputsFloat := trainDataset.EncodeToFloats(inputTokens...)
		copy(inputs.Data.GetFloats(), inputsFloat)
		pipeline.Forward()

		pos := len(inputTokens) - 1
		if pos < 0 {
			pos = 0
		}

		logits := output.Data.GetFloats()[pos*trainDataset.GetAlphabetSize() : (pos+1)*trainDataset.GetAlphabetSize()]
		f := sampleWithTemperature(logits, 0.9)
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

func sampleWithTemperature(logits []float32, temperature float32) int {
	for i := 0; i < len(logits); i++ {
		logits[i] /= temperature
	}

	return sampleFromDistribution(softmax(logits))
}

func softmax(logits []float32) []float32 {
	probs := make([]float32, len(logits))
	sum := 0.0

	for _, logit := range logits {
		sum += math.Exp(float64(logit))
	}

	for i, logit := range logits {
		probs[i] = float32(math.Exp(float64(logit)) / sum)
	}
	return probs
}

func sampleFromDistribution(probs []float32) int {
	r := float32(rand.Float64())
	cumulativeProb := float32(0.0)

	for i, prob := range probs {
		cumulativeProb += prob
		if r <= cumulativeProb {
			return i
		}
	}

	panic("unreachable code")
}
