package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/atkhx/nnet/examples/mnist/pkg"

	"github.com/atkhx/nnet/loss"

	"github.com/pkg/errors"

	"github.com/atkhx/nnet/dataset/mnist"
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

	fmt.Println("create convNet")
	convNet := pkg.CreateConvNet()
	if err = convNet.Init(); err != nil {
		return
	}

	fmt.Println("load convNet pretrain config from", nnetCfgFile)
	pretrainedConfig, err := os.ReadFile(nnetCfgFile)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return
	}

	if err != nil {
		fmt.Println("pretrain config not found", nnetCfgFile)
		fmt.Println("train your net before with make train")
		return
	}

	if err = json.Unmarshal(pretrainedConfig, convNet); err != nil {
		return
	}

	fmt.Println("load dataset")
	mnistDataset, err := mnist.CreateTestingDataset(datasetPath)
	if err != nil {
		return
	}

	lossFunction := loss.NewClassification()

	lossSum := 0.0
	success := 0
	statChunkSize := 1000
	sampleIndex := 0

	totalSuccess := 0
	totalLossSum := 0.0

	trainStopped := make(chan any)
	go func() {
		defer func() {
			fmt.Println()
			fmt.Println("convNet testing stopped")
			fmt.Println("- samples seen", sampleIndex)
			fmt.Println("- totalLossSum", totalLossSum)
			fmt.Println("- totalLossAvg", fmt.Sprintf("%.8f", totalLossSum/float64(sampleIndex)))
			fmt.Println("- totalSuccess", totalSuccess, "from", mnist.TestSetImagesCount)
			fmt.Println("- success rate", fmt.Sprintf("%.2f%%", 100*float64(totalSuccess)/float64(sampleIndex)))

			trainStopped <- true
		}()

		fmt.Println("convNet testing started")
		for sampleIndex = 0; sampleIndex < mnist.TestSetImagesCount; sampleIndex++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			input, target, err := mnistDataset.ReadSample(sampleIndex)
			if err != nil {
				return
			}

			output := convNet.Forward(input)

			outputIndex := output.GetMaxIndex()
			targetIndex := target.GetMaxIndex()

			if outputIndex == targetIndex {
				success++
				totalSuccess++
			}

			lossVal := lossFunction.GetError(target.Data, output.Data)
			lossSum += lossVal
			totalLossSum += lossVal

			if sampleIndex%statChunkSize == 0 {

				fmt.Println()
				fmt.Println("avg stat for samples", sampleIndex, "-", sampleIndex+statChunkSize)
				fmt.Println("- avg loss err\t", fmt.Sprintf("%.8f", lossSum/float64(statChunkSize)))
				fmt.Println("- success rate\t", fmt.Sprintf("%.2f%%", 100*float64(success)/float64(statChunkSize)))

				success = 0
				lossSum = 0.0
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
