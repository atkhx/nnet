package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/atkhx/nnet/gpt/dataset"
	"github.com/atkhx/nnet/gpt/pkg"
	numDevice "github.com/atkhx/nnet/num/dev/metal"
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

	device := numDevice.NewDevice()
	defer device.Release()

	batchSize := 1

	trainDataset := dataset.NewDataset(pkg.ContextLength, batchSize)
	trainDataset.ParseAlphabet()

	model := pkg.CreateModelForTest(
		trainDataset.GetAlphabetSize(),
		batchSize,
		device,
	)

	output := model.Compile()
	inputs := model.GetInput()

	if err := model.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := numDevice.NewPipeline(device, output)

	inputIndexes := trainDataset.EncodeString(`Китай`)
	inputTokens := trainDataset.Decode(inputIndexes...)

	fmt.Print(trainDataset.Decode(inputIndexes...))
	ctx := context.Background()
	for j := 0; j < pkg.ContextLength*10; j++ {
		inputsFloat := trainDataset.EncodeToFloats(inputTokens...)
		copy(inputs.Data.GetData(), inputsFloat)

		//commandBuffer := pipeline.GetCommandBuffer()
		//ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
		pipeline.Forward(ctx)
		//commandBuffer.Wait()

		pos := len(inputTokens) - 1
		if pos < 0 {
			pos = 0
		}

		logits := output.Data.GetData()[pos*trainDataset.GetAlphabetSize() : (pos+1)*trainDataset.GetAlphabetSize()]
		//numDevice.Float32s(logits).Softmax()       // probs
		//numDevice.Float32s(logits).CumulativeSum() // for Multinomial
		//f := numDevice.Float32s(logits).Multinomial()

		f := sampleWithTemperature(logits, 1)
		//f := greedyMax(logits)

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
