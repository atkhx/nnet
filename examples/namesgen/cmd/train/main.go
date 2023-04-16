package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atkhx/nnet/examples/namesgen/dataset"
	"github.com/atkhx/nnet/examples/namesgen/pkg"
	"github.com/atkhx/nnet/loss"
)

const (
	epochsCount = 1
)

var filename string

func init() {
	rand.Seed(time.Now().UnixNano())

	flag.StringVar(&filename, "c", "./examples/namesgen/config.json", "nn config file")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	namesDataset := dataset.NewDataset(dataset.NamesContextSize, dataset.NamesMiniBatchSize, dataset.Names)

	seqModel := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		namesDataset.GetContextSize(),
		namesDataset.GetMiniBatchSize(),
	)

	seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	defer func() {
		if err := seqModel.SaveToFile(filename); err != nil {
			log.Fatalln(err)
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	trainStopped := make(chan any)
	go func() {
		defer func() {
			trainStopped <- true
		}()

		statChunkSize := 1000
		lossAvg := 0.0

		output := seqModel.NewOutput()

		for epoch := 0; epoch < epochsCount; epoch++ {
			for sampleIndex := 0; sampleIndex < namesDataset.GetSamplesCount(); sampleIndex++ {
				select {
				case <-ctx.Done():
					return
				default:
				}

				batchInputs, batchTarget := namesDataset.ReadRandomSampleBatch()

				seqModel.Forward(batchInputs, output)

				lossAvg += loss.RegressionMean(dataset.NamesMiniBatchSize, batchTarget, output)

				seqModel.Backward(batchTarget)
				seqModel.Update(0.1)

				if sampleIndex > 0 && sampleIndex%statChunkSize == 0 {
					lossAvg /= float64(statChunkSize)
					fmt.Println("loss", lossAvg)
					lossAvg = 0
				}
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
