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

	"github.com/pkg/errors"

	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/examples/mnist/pkg"
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
	dataset, err := mnist.CreateTestingDataset(datasetPath)
	if err != nil {
		return
	}

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
			fmt.Println("- totalSuccess", totalSuccess, "from", dataset.GetSamplesCount())
			fmt.Println("- success rate", fmt.Sprintf("%.2f%%", 100*float64(totalSuccess)/float64(sampleIndex)))

			trainStopped <- true
		}()

		fmt.Println("convNet testing started")
		for sampleIndex = 0; sampleIndex < dataset.GetSamplesCount(); sampleIndex++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			input, target, err := dataset.ReadSample(sampleIndex)
			if err != nil {
				return
			}

			output := convNet.Forward(input)

			_, outputIndex := output.Data.GetMax()
			//outputIndex := getMaxIndex(output)
			_, targetIndex := target.Data.GetMax()

			if outputIndex == targetIndex {
				success++
				totalSuccess++
			}

			lossVal := output.Classification(target).Mean().Data.Data[0]
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
