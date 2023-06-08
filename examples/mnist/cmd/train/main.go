package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/examples/mnist/pkg"
	"github.com/atkhx/nnet/num"
)

const (
	epochsCount = 100_000
	batchSize   = 1
)

var (
	datasetPath string
	nnetCfgFile string
)

func init() {
	flag.StringVar(&datasetPath, "d", "./examples/mnist/data/", "path to dataset files")
	flag.StringVar(&nnetCfgFile, "c", "./examples/mnist/config.json", "convNet config file")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())

	dataset, err := mnist.CreateTrainingDataset(datasetPath)
	if err != nil {
		return
	}

	seqModel := pkg.CreateConvNet(batchSize)

	modelOutput := seqModel.Compile()
	if err := seqModel.LoadFromFile(nnetCfgFile); err != nil {
		log.Fatalln(err)
	}

	defer func() {
		if err := seqModel.SaveToFile(nnetCfgFile); err != nil {
			log.Fatalln(err)
		}
	}()

	targets := num.New(num.NewDims(10, 1, batchSize))

	loss := modelOutput.CrossEntropy(targets)
	lossMean := loss.Mean()

	forwardNodes := lossMean.GetForwardNodes()
	backwardNodes := lossMean.GetBackwardNodes()
	resetGradsNodes := lossMean.GetResetGradsNodes()
	fmt.Println("forwardNodes", len(forwardNodes))
	fmt.Println("backwardNodes", len(backwardNodes))
	fmt.Println("resetGradsNodes", len(resetGradsNodes))

	fmt.Println("trainable params count:", seqModel.GetTrainableParamsCount())

	lossAvg := 0.0
	statChunkSize := 100

	trainStopped := make(chan any)
	go func() {
		defer func() {
			trainStopped <- true
		}()

		t := time.Now()
		for index := 0; index < epochsCount; index++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			sample, err := dataset.ReadRandomSampleBatch(batchSize)
			if err != nil {
				log.Fatalln(err)
			}
			copy(targets.Data, sample.Target.Data)
			copy(seqModel.GetInput().Data, sample.Input.Data)

			forwardNodes.Forward()
			resetGradsNodes.ResetGrad()
			backwardNodes.Backward()

			lossAvg += lossMean.Data[0]
			seqModel.Update()

			if index > 0 && index%statChunkSize == 0 {
				lossAvg /= float64(statChunkSize)
				fmt.Println(
					fmt.Sprintf("loss: %.8f", lossAvg), "\t",
					"index:", index, "\t",
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
