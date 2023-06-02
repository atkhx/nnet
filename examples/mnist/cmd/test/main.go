package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/examples/mnist/pkg"
	"github.com/atkhx/nnet/num"
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

	seqModel := pkg.CreateConvNet(1)

	modelOutput := seqModel.Compile()
	if err := seqModel.LoadFromFile(nnetCfgFile); err != nil {
		log.Fatalln(err)
	}

	fmt.Println("load dataset")
	dataset, err := mnist.CreateTestingDataset(datasetPath)
	if err != nil {
		return
	}

	targets := num.New(num.NewDims(10, 1, 1))

	loss := modelOutput.CrossEntropy(targets)

	fmt.Println("trainable params count:", seqModel.GetTrainableParamsCount())

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

			inputs, target, err := dataset.ReadSample(sampleIndex)
			if err != nil {
				log.Fatalln(err)
			}
			copy(targets.Data, target.Data)
			seqModel.Forward(inputs.Data)

			loss.Forward()

			outputIndex, _ := modelOutput.Data.MaxKeyVal()
			targetIndex, _ := target.Data.MaxKeyVal()

			if outputIndex == targetIndex {
				success++
				totalSuccess++
			}

			lossVal := loss.Data[0]
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
