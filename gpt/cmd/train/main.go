package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atkhx/nnet/gpt/dataset"
	"github.com/atkhx/nnet/gpt/pkg"
	"github.com/atkhx/nnet/num"
	numDevice "github.com/atkhx/nnet/num/dev/metal"

	"github.com/dustin/go-humanize"
)

var (
	filename string
	statSize int
	epochs   int
)

func init() {
	flag.StringVar(&filename, "c", "./gpt/config.json", "nn config file")
	flag.IntVar(&epochs, "i", 5_000, "iterations count")
	flag.IntVar(&statSize, "s", 10, "show statistics every n iterations")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	go func() {
		log.Println(http.ListenAndServe(":6060", nil))
	}()

	device := numDevice.NewDevice()
	defer device.Close()

	trainDataset := dataset.NewDataset(pkg.ContextLength, pkg.MiniBatchSize)
	trainDataset.ParseAlphabet()
	trainDataset.ParseTokens()

	optimizer := pkg.CreateOptimizer(epochs, device)
	model := pkg.CreateModel(trainDataset.GetAlphabetSize(), pkg.MiniBatchSize, device, optimizer)

	output := model.Compile()
	inputs := model.GetInput()
	target := device.NewData(num.NewDims(1, pkg.ContextLength*pkg.MiniBatchSize))

	lossFunc := device.CrossEntropyPos(output, target)
	lossMean := device.Mean(lossFunc)
	pipeline := numDevice.NewPipeline(device, lossMean)

	if err = model.LoadFromFile(filename); err != nil {
		return
	}

	fmt.Println("params weight:", humanize.Bytes(uint64(model.GetTrainableParamsCount()*4)))
	fmt.Println("params count:", model.GetTrainableParamsCount())
	fmt.Println("alphabet size:", trainDataset.GetAlphabetSize())
	fmt.Println("features count:", pkg.FeaturesCount)

	defer func() {
		if err := model.SaveToFile(filename); err != nil {
			log.Fatalln(err)
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	trainStopped := make(chan any)

	go func() {
		defer close(trainStopped)

		lossAvg := float32(0.0)

		fmt.Println("training started")
		t := time.Now()

		for iteration := 0; iteration < epochs; iteration++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			batchInputs, batchTarget := trainDataset.ReadRandomSampleBatch()
			copy(target.Data, batchTarget)
			copy(inputs.Data, batchInputs)

			pipeline.Forward(ctx)
			pipeline.Backward(ctx)
			model.Update(iteration)

			lossAvg += lossMean.Data[0]

			//if iteration == 0 {
			//	generateSample(ctx, trainDataset, pipeline, inputs, output)
			//}

			if iteration > 0 && iteration%statSize == 0 {
				lossAvg /= float32(statSize)
				fmt.Println(
					fmt.Sprintf("lossFunc: %.8f", lossAvg), "\t",
					"iteration:", iteration, "\t",
					"duration:", time.Since(t), "\t",
				)
				lossAvg = 0

				//if iteration%(10*statSize) == 0 {
				//	generateSample(ctx, trainDataset, pipeline, inputs, output)
				//}

				t = time.Now()
			}
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-trainStopped:
	case <-quit:
		cancel()
		<-trainStopped
	}
}

func generateSample(
	ctx context.Context,
	trainDataset *dataset.Dataset,
	pipeline *numDevice.Pipeline,
	inputs *num.Data,
	output *num.Data,
) {
	fmt.Println("generate sample")
	defer fmt.Println()

	inputIndexes := trainDataset.EncodeString(` `)
	inputTokens := trainDataset.Decode(inputIndexes...)

	fmt.Print(trainDataset.Decode(inputIndexes...))
	for j := 0; j < pkg.ContextLength; j++ {
		inputsFloat := trainDataset.EncodeToFloats(inputTokens...)
		copy(inputs.Data, inputsFloat)

		pipeline.Forward(ctx)

		pos := len(inputTokens) - 1
		if pos < 0 {
			pos = 0
		}

		logits := output.Data[pos*trainDataset.GetAlphabetSize() : (pos+1)*trainDataset.GetAlphabetSize()]
		numDevice.Float32s(logits).Softmax() // probs
		numDevice.Float32s(logits).CumulativeSum()
		f := numDevice.Float32s(logits).Multinomial()
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
