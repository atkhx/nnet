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

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/atkhx/nnet/examples/ts-sa/dataset"
	"github.com/atkhx/nnet/examples/ts-sa/metrics"
	"github.com/atkhx/nnet/examples/ts-sa/pkg"
	"github.com/atkhx/nnet/num"
)

var filename string
var epochs int

func init() {
	flag.StringVar(&filename, "c", "./examples/ts-sa/config.json", "nn config file")
	flag.IntVar(&epochs, "e", 5_000, "epochs count")
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

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		log.Println(http.ListenAndServe(":8181", nil))
	}()

	namesDataset := dataset.NewDataset(pkg.ContextLength, pkg.MiniBatchSize)
	namesDataset.ParseAlphabet(dataset.TinyShakespeare)

	seqModel := pkg.CreateNN(namesDataset.GetAlphabetSize(), pkg.MiniBatchSize)

	modelOutput := seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	targets := num.New(num.NewDims(1, pkg.ContextLength*pkg.MiniBatchSize))
	lossMean := modelOutput.CrossEntropyPos(targets).Mean()
	pipeline := num.NewPipeline(lossMean)

	fmt.Println("trainable params count:", seqModel.GetTrainableParamsCount())

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

		statChunkSize := 10
		lossAvg := 0.0

		t := time.Now()

		for index := 0; index < epochs; index++ {
			t1 := time.Now()
			select {
			case <-ctx.Done():
				return
			default:
			}

			batchInputs, batchTarget := namesDataset.ReadRandomSampleBatch()
			copy(targets.Data, batchTarget)

			//seqModel.Forward(batchInputs)
			copy(seqModel.GetInput().Data, batchInputs)

			pipeline.Forward()
			pipeline.Backward()

			metrics.LossMean.Set(lossMean.Data[0])
			metrics.TrainDuration.Add(float64(time.Since(t1).Milliseconds()))

			lossAvg += lossMean.Data[0]
			seqModel.Update()

			if index > 0 && index%statChunkSize == 0 {
				lossAvg /= float64(statChunkSize)
				metrics.AvgLossMean.Set(lossAvg)
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
