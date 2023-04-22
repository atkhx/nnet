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

	"github.com/atkhx/nnet/examples/ts/dataset"
	"github.com/atkhx/nnet/examples/ts/pkg"
	"github.com/atkhx/nnet/loss"
)

var filename string
var epochs int

func init() {
	rand.Seed(time.Now().UnixNano())

	flag.StringVar(&filename, "c", "./examples/ts/config.json", "nn config file")
	flag.IntVar(&epochs, "e", 5_000_000, "epochs count")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	namesDataset := dataset.NewDataset(dataset.NamesContextSize, dataset.NamesMiniBatchSize)
	namesDataset.ParseAlphabet(dataset.TinyShakespeare)

	//inputs, targets := namesDataset.ReadRandomSampleBatch()
	//fmt.Println(string(namesDataset.DecodeFloats(inputs...)))
	//fmt.Println(targets)

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

		for index := 0; index < epochs; index++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			batchInputs, batchTarget := namesDataset.ReadRandomSampleBatch()

			seqModel.Forward(batchInputs, output)

			lossAvg += loss.CrossEntropy(batchTarget, output, namesDataset.GetMiniBatchSize())
			oGrads := loss.CrossEntropyBackward(batchTarget, output, namesDataset.GetMiniBatchSize())

			seqModel.Backward(oGrads)
			seqModel.Update(0.1)

			if index > 0 && index%statChunkSize == 0 {
				lossAvg /= float64(statChunkSize)
				fmt.Println("loss", lossAvg)
				lossAvg = 0
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
