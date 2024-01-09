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

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/proc"
	"github.com/atkhx/nnet/gpt/dataset"
	"github.com/atkhx/nnet/gpt/pkg"
	"github.com/dustin/go-humanize"
)

var (
	filename string
	statSize int
	epochs   int
)

func init() {
	flag.StringVar(&filename, "c", "./gpt/config.json", "nn config file")
	flag.IntVar(&epochs, "i", pkg.TrainingIterations, "iterations count")
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

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	trainDataset := dataset.NewDataset(pkg.ContextLength, pkg.MiniBatchSize)
	trainDataset.ParseAlphabet()
	trainDataset.ParseTokens()

	optimizer := pkg.CreateOptimizer(epochs, device)
	model := pkg.CreateTrainingModel(trainDataset.GetAlphabetSize(), pkg.MiniBatchSize, device, optimizer)

	output := model.Compile()
	inputs := model.GetInput()
	target := device.NewData(mtl.NewMTLSize(1, pkg.ContextLength*pkg.MiniBatchSize))

	lossFunc := device.CrossEntropyPos(output, target)
	lossMean := device.Mean(lossFunc)
	pipeline := device.GetTrainingPipeline(lossMean)

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
			copy(target.Data.GetFloats(), batchTarget)
			copy(inputs.Data.GetFloats(), batchInputs)

			pipeline.TrainIteration(func(b *mtl.CommandBuffer) {
				model.Update(b, iteration)
			})

			lossAvg += lossMean.Data.GetFloats()[0]
			if (iteration > 0 || statSize == 1) && iteration%statSize == 0 {
				lossAvg /= float32(statSize)
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
