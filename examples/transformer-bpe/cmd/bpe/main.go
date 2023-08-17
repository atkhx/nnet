package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

var (
	modelName string
)

func init() {
	flag.StringVar(&modelName, "model", "bert-base-uncased", "model name as at Huggingface model hub e.g. 'tiiuae/falcon-7b'. Default='bert-base-uncased'")
}

func main() {
	flag.Parse()

	// any model with file `tokenizer.json` available. Eg. `tiiuae/falcon-7b`, `TheBloke/guanaco-7B-HF`, `mosaicml/mpt-7b-instruct`
	// configFile, err := tokenizer.CachedPath("bert-base-uncased", "tokenizer.json")
	configFile, err := tokenizer.CachedPath(modelName, "tokenizer.json")
	if err != nil {
		panic(err)
	}

	tk, err := pretrained.FromFile(configFile)
	if err != nil {
		panic(err)
	}

	sentence := `The Gophers craft code using [MASK] language.`
	en, err := tk.EncodeSingle(sentence, true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%-10s: %q\n", "Tokens", en.Tokens)
	fmt.Printf("%-10s: %v\n", "Ids", en.Ids)
	fmt.Printf("%-10s: %v\n", "Offsets", en.Offsets)

	vocab := tk.GetVocab(true)

	fmt.Println("len(vocab)", len(vocab))

	vocabIndex := make(map[int]struct{}, len(vocab))
	for _, i := range vocab {
		vocabIndex[i] = struct{}{}
	}

	for i := 0; i < len(vocab); i++ {
		if _, ok := vocabIndex[i]; !ok {
			log.Println("token index not found", i)
			break
		}
	}

	idx := []int{101, 1996, 2175, 27921, 2015, 7477, 3642, 2478, 103, 2653, 1012, 102}

	fmt.Println(tk.Decode(idx, false))

	for i := 1; i < len(idx); i++ {
		fmt.Println(tk.Decode(idx[:i], false))
	}

}
