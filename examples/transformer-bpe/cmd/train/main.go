package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atkhx/nnet/examples/transformer-bpe/dataset"
	"github.com/atkhx/nnet/examples/transformer-bpe/pkg"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/optimizer"
	//"github.com/sugarme/tokenizer"
	"github.com/cohere-ai/tokenizer"
)

var (
	filename      string
	statChunkSize int
	epochs        int
	modelName     string
)

func init() {
	flag.StringVar(&filename, "c", "./examples/transformer-bpe/config.json", "nn config file")
	flag.StringVar(&modelName, "model", "bert-base-uncased", "model name as at Huggingface model hub e.g. 'tiiuae/falcon-7b'. Default='bert-base-uncased'")
	flag.IntVar(&epochs, "i", 5_000, "iterations count")
	flag.IntVar(&statChunkSize, "s", 10, "show statistics every n iterations")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	go func() {
		log.Println(http.ListenAndServe(":6060", nil))
	}()

	//configFile, err := tokenizer.CachedPath(modelName, "tokenizer.json")
	//if err != nil {
	//	panic(err)
	//}
	//
	//tk, err := pretrained.FromFile(configFile)
	//if err != nil {
	//	panic(err)
	//}

	encoder, err := tokenizer.NewFromPrebuilt("coheretext-50k")
	if err != nil {
		log.Fatal(err)
	}

	trainingFilename := "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/transformer-bpe/data/ruwiki1.txt"
	//trainingFilename := "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/transformer-bpe/src/tiny.txt"
	trainDataset := dataset.NewDataset(encoder, pkg.ContextLength, pkg.TrainingMiniBatchSize)
	if err := trainDataset.OpenTrainingFile(trainingFilename); err != nil {
		panic(err)
	}

	modelOptimizer := optimizer.NewOptimizerAdam(epochs, 0.9, 0.98, 3e-4, 0.000000001)
	model := pkg.CreateNN(trainDataset.GetAlphabetSize(), pkg.TrainingMiniBatchSize, modelOptimizer)

	modelOutput := model.Compile()
	if err := model.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	targets := num.New(num.NewDims(1, pkg.ContextLength*pkg.TrainingMiniBatchSize))
	inputs := model.GetInput()

	lossFunc := modelOutput.CrossEntropyPos(targets)
	pipeline := num.NewPipeline(lossFunc)
	lossMean := lossFunc.Mean()

	fmt.Println("trainable params count:", model.GetTrainableParamsCount())
	fmt.Println("alphabet size:", trainDataset.GetAlphabetSize())

	defer func() {
		if err := model.SaveToFile(filename); err != nil {
			log.Fatalln(err)
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	trainStopped := make(chan any)

	go func() {
		defer close(trainStopped)

		lossAvg := 0.0

		t := time.Now()

		for iteration := 0; iteration < epochs; iteration++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			batchInputs, batchTarget := trainDataset.ReadRandomSampleBatch()
			copy(targets.Data, batchTarget)
			copy(inputs.Data, batchInputs)

			pipeline.Forward()
			lossMean.Forward()
			pipeline.Backward()
			model.Update(iteration)

			lossAvg += lossMean.Data[0]

			if iteration > 0 && iteration%statChunkSize == 0 {
				lossAvg /= float64(statChunkSize)
				fmt.Println(
					fmt.Sprintf("lossFunc: %.8f", lossAvg), "\t",
					"iteration:", iteration, "\t",
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
