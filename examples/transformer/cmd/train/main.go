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

	"github.com/atkhx/mps"
	"github.com/atkhx/nnet/examples/transformer/dataset"
	"github.com/atkhx/nnet/examples/transformer/pkg"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/num/native"
)

var (
	filename      string
	statChunkSize int
	epochs        int
)

func init() {
	flag.StringVar(&filename, "c", "./examples/transformer/config.json", "nn config file")
	flag.IntVar(&epochs, "i", 5_000, "iterations count")
	flag.IntVar(&statChunkSize, "s", 10, "show statistics every n iterations")
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

	mps.InitDefaultDevice()
	defer mps.ReleaseDefaultDevice()

	trainDataset := dataset.NewDataset(pkg.ContextLength, pkg.TrainingMiniBatchSize)
	trainDataset.ParseAlphabet()
	trainDataset.ParseTokens()

	device := &native.Device{}

	//modelOptimizer := device.GetOptimizerAdam(epochs, 0.9, 0.98, 3e-4, 0.000000001)
	//modelOptimizer := device.GetOptimizerAdam(epochs, 0.9, 0.98, 0.0003, 0.000000001)
	modelOptimizer := device.GetOptimizerAdam(epochs, 0.9, 0.98, 0.0003, 0.000000001)
	model := pkg.CreateNN[*native.Data](trainDataset.GetAlphabetSize(), pkg.TrainingMiniBatchSize, device, modelOptimizer)

	modelOutput := model.Compile()
	if err := model.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	targets := device.NewData(num.NewDims(1, pkg.ContextLength*pkg.TrainingMiniBatchSize))
	inputs := model.GetInput()

	lossFunc := device.CrossEntropyPos(modelOutput, targets)
	//pipeline := native.NewPipeline(lossFunc)
	lossMean := device.Mean(lossFunc)
	pipeline := native.NewPipeline(lossMean)

	fmt.Println("trainable params count:", model.GetTrainableParamsCount())
	fmt.Println("alphabet size:", trainDataset.GetAlphabetSize())

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

		t := time.Now()

		for iteration := 0; iteration < epochs; iteration++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			batchInputs, batchTarget := trainDataset.ReadRandomSampleBatch()
			copy(targets.Data, batchTarget)
			copy(inputs.Data, batchInputs)

			pipeline.Forward(ctx)
			//lossMean.Forward(ctx)
			pipeline.Backward(ctx)
			model.Update(iteration)

			lossAvg += lossMean.Data[0]

			if iteration > 0 && iteration%statChunkSize == 0 {
				lossAvg /= float32(statChunkSize)
				fmt.Println(
					fmt.Sprintf("lossFunc: %.8f", lossAvg), "\t",
					"iteration:", iteration, "\t",
					"duration:", time.Since(t), "\t",
				)
				lossAvg = 0
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
