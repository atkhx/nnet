package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/pkg/errors"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/examples/namesgen/dataset"
	"github.com/atkhx/nnet/examples/namesgen/pkg"
)

var (
	nnetCfgFile string
)

func init() {
	flag.StringVar(&nnetCfgFile, "c", "./examples/namesgen/config.json", "nn config file")
	flag.Parse()
}

func main() {
	rand.Seed(time.Now().UnixNano())

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
		fmt.Println("train your net before with make train")
		return
	}

	if err = json.Unmarshal(pretrainedConfig, nn); err != nil {
		return
	}

	fmt.Println()

	for j := 0; j < 100; j++ {
		inputBytes := bytes.Repeat([]byte{'.'}, namesDataset.GetContextSize())
		for k := 0; k < 50; k++ {
			output := nn.Forward(data.NewOneHotVectors(
				namesDataset.GetAlphabetSize(), namesDataset.Encode(inputBytes...)...,
			))

			b := namesDataset.Decode(data.Multinomial(output.Data.Softmax().Data))

			inputBytes = append(inputBytes, b...)
			inputBytes = inputBytes[len(b):]

			if b[len(b)-1] == '.' {
				//fmt.Println(string(inputBytes))
				fmt.Println()
				break
			}
			fmt.Print(string(b))
		}
	}

	fmt.Println()
}
