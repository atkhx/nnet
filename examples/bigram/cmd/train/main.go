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

	data2 "github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/examples/bigram/data"
	"github.com/atkhx/nnet/examples/bigram/pkg"
	"github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
)

const (
	batchSize   = 1
	epochsCount = 1
)

var (
	nnetCfgFile string
)

func init() {
	flag.StringVar(&nnetCfgFile, "c", "./examples/bigram/config.json", "nn config file")
	flag.Parse()
}

func main() {
	//fmt.Println(math.Log(math.E))
	//return
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())

	fmt.Println("create nn")
	nn := pkg.CreateNN(data.AlphabetSize, data.WordLen)

	fmt.Println("load nn pretrain config from", nnetCfgFile)
	pretrainedConfig, err := os.ReadFile(nnetCfgFile)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return
	}

	if err != nil {
		fmt.Println("pretrain config not found", nnetCfgFile)
		fmt.Println("will be created new one on finish")
	} else if err = json.Unmarshal(pretrainedConfig, nn); err != nil {
		return
	}

	fmt.Println("create trainer")
	netTrainer := trainer.New(
		nn,
		//trainer.WithMethod(methods.VanilaSGD(0.1)),
	)

	var sampleIndex int
	defer func() {
		fmt.Println("save nn pretrain config to", nnetCfgFile)
		if err = saveNet(nn, nnetCfgFile); err != nil {
			return
		}
	}()

	lossSum := 0.0
	statChunkSize := 1000
	totalLossSum := 0.0

	trainStopped := make(chan any)
	go func() {
		defer func() {
			fmt.Println()
			fmt.Println("nn training stopped")
			fmt.Println("- samples seen", sampleIndex)
			fmt.Println("- totalLossSum", totalLossSum/float64(len(data.Samples)))
			fmt.Println("- totalLossAvg", fmt.Sprintf("%.8f", totalLossSum/float64(sampleIndex)))

			trainStopped <- true
		}()

		fmt.Println("nn training started")
		statChunkSize = 1000
		for epoch := 0; epoch < epochsCount; epoch++ {
			//for sampleIndex = 0; sampleIndex < len(data.Samples); sampleIndex++ {
			for sampleIndex = 0; sampleIndex < 100_000; sampleIndex++ {
				select {
				case <-ctx.Done():
					return
				default:
				}

				sample := data.Samples[sampleIndex]

				lossObject := netTrainer.Forward(sample.Input, func(output *data2.Matrix) *data2.Matrix {
					return output.Classification(sample.Target).Mean()
				})

				loss := lossObject.Data[0]
				lossSum += loss
				totalLossSum += loss

				if sampleIndex > 0 && sampleIndex%statChunkSize == 0 {
					//fmt.Println()
					//fmt.Println("avg stat for samples", sampleIndex, "-", sampleIndex+statChunkSize)
					fmt.Println("- avg loss err\t", fmt.Sprintf("%.8f", lossSum/float64(statChunkSize)))
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

func saveNet(
	nn *net.FeedForward,
	configFilename string,
) error {
	nnBytes, err := json.Marshal(nn)
	if err != nil {
		return errors.Wrap(err, "marshal nn config failed")
	}

	if err := os.WriteFile(configFilename, nnBytes, os.ModePerm); err != nil {
		return errors.Wrap(err, "write nn config failed")
	}
	return nil
}
