package main

import (
	"fmt"
	"log"

	"github.com/cohere-ai/tokenizer"
)

func main() {
	encoder, err := tokenizer.NewFromPrebuilt("coheretext-50k")
	if err != nil {
		log.Fatal(err)
	}
	ii, ss := encoder.Encode("Fuck you!")
	fmt.Println("ii", ii)
	fmt.Println("ss", ss)

	fmt.Println(encoder.Decode(ii))
}
