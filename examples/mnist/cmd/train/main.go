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

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/examples/mnist/pkg"
	"github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
)

const (
	epochsCount = 1
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
		fmt.Println("will be created new one on finish")
	} else if err = json.Unmarshal(pretrainedConfig, convNet); err != nil {
		return
	}

	fmt.Println("load dataset")
	dataset, err := mnist.CreateTrainingDataset(datasetPath)
	if err != nil {
		return
	}

	fmt.Println("create trainer")
	netTrainer := trainer.New(convNet)

	var sampleIndex int
	//defer func() {
	//	fmt.Println("save convNet pretrain config to", nnetCfgFile)
	//	if err = saveConvNet(convNet, nnetCfgFile); err != nil {
	//		return
	//	}
	//}()

	lossSum := 0.0
	//success := 0
	statChunkSize := 1000

	totalLossSum := 0.0
	totalSuccess := 0

	trainStopped := make(chan any)
	go func() {
		defer func() {
			fmt.Println()
			fmt.Println("convNet training stopped")
			fmt.Println("- samples seen", sampleIndex)
			fmt.Println("- totalLossSum", totalLossSum)
			fmt.Println("- totalLossAvg", fmt.Sprintf("%.8f", totalLossSum/float64(sampleIndex)))
			fmt.Println("- totalSuccess", totalSuccess, "from", dataset.GetSamplesCount())
			fmt.Println("- success rate", fmt.Sprintf("%.2f%%", 100*float64(totalSuccess)/float64(sampleIndex)))

			trainStopped <- true
		}()

		statChunkSize = 100
		fmt.Println("convNet training started")
		for epoch := 0; epoch < epochsCount; epoch++ {
			for sampleIndex = 0; sampleIndex < dataset.GetSamplesCount(); sampleIndex++ {
				//for sampleIndex = 0; sampleIndex < 2000; sampleIndex++ {
				select {
				case <-ctx.Done():
					return
				default:
				}

				input, target, err := dataset.ReadSample(sampleIndex)
				if err != nil {
					return
				}

				//fmt.Println("input", input.GetDims())
				//fmt.Println("input", input.Data)

				lossObject := netTrainer.Forward(input, func(output *data.Data) *data.Data {
					loss := output.Classification(target).Mean()
					return loss
				})

				loss := lossObject.Data.Data[0]
				lossSum += loss
				totalLossSum += loss

				if sampleIndex%statChunkSize == 0 {
					fmt.Println("lossObject", lossObject.Data)
					//fmt.Println()
					//fmt.Println("avg stat for samples", sampleIndex, "-", sampleIndex+statChunkSize)
					//fmt.Println("- avg loss err\t", fmt.Sprintf("%.8f", lossSum/float64(statChunkSize)))
					//fmt.Println("- success rate\t", fmt.Sprintf("%.2f%%", 100*float64(success)/float64(statChunkSize)))

					//success = 0
					lossSum = 0.0
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

func saveConvNet(
	convNet *net.FeedForward,
	configFilename string,
) error {
	convNetBytes, err := json.Marshal(convNet)
	if err != nil {
		return errors.Wrap(err, "marshal convNet config failed")
	}

	if err := os.WriteFile(configFilename, convNetBytes, os.ModePerm); err != nil {
		return errors.Wrap(err, "write convNet config failed")
	}
	return nil
}
