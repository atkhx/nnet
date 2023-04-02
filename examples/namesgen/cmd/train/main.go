package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/pkg/errors"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/examples/namesgen/dataset"
	"github.com/atkhx/nnet/examples/namesgen/pkg"
	"github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
)

const (
	epochsCount = 1
)

var nnetCfgFile string

func init() {
	rand.Seed(time.Now().UnixNano())

	flag.StringVar(&nnetCfgFile, "c", "./examples/namesgen/config.json", "nn config file")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	fmt.Println("initialize names dataset")
	fmt.Println("- contextSize:", dataset.NamesContextSize)
	fmt.Println("- miniBatchSize:", dataset.NamesMiniBatchSize)

	namesDataset := dataset.NewDataset(dataset.NamesContextSize, dataset.NamesMiniBatchSize, dataset.Names)

	ctx, cancel := context.WithCancel(context.Background())

	fmt.Println("create nn")
	nn := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		namesDataset.GetContextSize(),
		namesDataset.GetMiniBatchSize(),
	)

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
	netTrainer := trainer.New(nn)

	var sampleIndex int
	defer func() {
		fmt.Println("save nn pretrain config to", nnetCfgFile)
		if err = saveNet(nn, nnetCfgFile); err != nil {
			return
		}
	}()

	statChunkSize := 1000

	fmt.Println("show statistics every", statChunkSize, "iteration")

	lossSum := 0.0
	totalLossSum := 0.0

	trainStopped := make(chan any)
	go func() {
		defer func() {
			fmt.Println()
			fmt.Println("nn training stopped")
			fmt.Println("- samples seen", sampleIndex)
			fmt.Println("- totalLossSum", totalLossSum/float64(namesDataset.GetSamplesCount()))
			fmt.Println("- totalLossAvg", fmt.Sprintf("%.8f", totalLossSum/float64(sampleIndex)))

			trainStopped <- true
		}()

		fmt.Println("nn training started")
		for epoch := 0; epoch < epochsCount; epoch++ {
			for sampleIndex = 0; sampleIndex < namesDataset.GetSamplesCount(); sampleIndex++ {
				select {
				case <-ctx.Done():
					return
				default:
				}

				input, target, _ := namesDataset.ReadRandomSampleBatch()
				lossObject := netTrainer.Forward(input, func(output *data.Data) *data.Data {
					return output.CrossEntropy(target).Mean()
				})

				loss := lossObject.Data.Data[0]
				lossSum += loss
				totalLossSum += loss

				if sampleIndex > 0 && sampleIndex%statChunkSize == 0 {
					fmt.Println(
						"",
						"\t- avg loss err\t", fmt.Sprintf("%.8f", lossSum/float64(statChunkSize)),
						"\t - last loss\t", fmt.Sprintf("%.8f", loss),
					)
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
