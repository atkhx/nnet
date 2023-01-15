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
	"github.com/atkhx/nnet/loss"
	"github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
	"github.com/atkhx/nnet/trainer/methods"
)

const (
	batchSize    = 10
	learningRate = 0.01
	epochsCount  = 1
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
		fmt.Println("will be created new one on finish")
	} else if err = json.Unmarshal(pretrainedConfig, convNet); err != nil {
		return
	}

	fmt.Println("load dataset")
	mnistDataset, err := mnist.CreateTrainingDataset(datasetPath)
	if err != nil {
		return
	}

	fmt.Println("create trainer")
	mnistTrainer := trainer.New(
		convNet,
		//methods.Adadelta(trainer.Ro, trainer.Eps),
		methods.Adagard(learningRate, trainer.Eps),
		//methods.Nesterov(0.01, 0.9),
		batchSize,
	)

	lossFunction := loss.NewClassification()

	var sampleIndex int
	defer func() {
		fmt.Println("save convNet pretrain config to", nnetCfgFile)
		if err = saveConvNet(convNet, nnetCfgFile); err != nil {
			return
		}
	}()

	lossSum := 0.0
	success := 0
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
			fmt.Println("- totalSuccess", totalSuccess, "from", mnist.TrainSetImagesCount)
			fmt.Println("- success rate", fmt.Sprintf("%.2f%%", 100*float64(totalSuccess)/float64(sampleIndex)))

			trainStopped <- true
		}()

		fmt.Println("convNet training started")
		for epoch := 0; epoch < epochsCount; epoch++ {
			for sampleIndex = 0; sampleIndex < mnist.TrainSetImagesCount; sampleIndex++ {
				select {
				case <-ctx.Done():
					return
				default:
				}

				input, target, err := mnistDataset.ReadSample(sampleIndex)
				if err != nil {
					return
				}

				output := mnistTrainer.Forward(input, target)
				mnistTrainer.UpdateWeights()

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

	f, err := os.OpenFile(configFilename, os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		return errors.Wrap(err, "create file for convNet config failed")
	}
	defer func() {
		_ = f.Close()
	}()

	if _, err = f.Write(convNetBytes); err != nil {
		return errors.Wrap(err, "write convNet config failed")
	}
	return nil
}
