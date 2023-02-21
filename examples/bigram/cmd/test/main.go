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

	data2 "github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/examples/bigram/data"
	"github.com/atkhx/nnet/examples/bigram/pkg"
	"github.com/atkhx/nnet/floats"
)

var (
	nnetCfgFile string
)

func init() {
	flag.StringVar(&nnetCfgFile, "c", "./examples/bigram/config.json", "nn config file")
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

	fmt.Println("create nn")
	nn := pkg.CreateNN(data.AlphabetSize, data.WordLen)
	nn.Init()

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
		inputBytes := bytes.Repeat([]byte{'.'}, data.WordLen)
		for {
			output := nn.Forward(data2.MustCompileMultiOneHotsMatrix(
				data.AlphabetSize, data.Encode(inputBytes)...,
			))

			b := data.Chars[floats.Multinomial(output.Data)]
			fmt.Print(string(b))

			inputBytes = append(inputBytes, b)
			inputBytes = inputBytes[1:]

			if b == '.' {
				fmt.Println()
				break
			}
		}
	}

	fmt.Println()
}
