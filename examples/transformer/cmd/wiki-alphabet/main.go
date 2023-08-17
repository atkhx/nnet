package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
)

var fIn = "./examples/transformer/data/ruwiki12.txt"
var fOut = "./examples/transformer/data/ruwiki12.alphabet"

func main() {
	fileIn, err := os.OpenFile(fIn, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer fileIn.Close()

	fileOut, err := os.OpenFile(fOut, os.O_CREATE|os.O_RDWR, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer fileOut.Close()

	reader := bufio.NewReader(fileIn)

	seen := map[rune]struct{}{}
	runes := []rune{}

	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			log.Println(err)
			break
		}

		if _, ok := seen[r]; !ok {
			seen[r] = struct{}{}
			runes = append(runes, r)
		}
	}

	fmt.Println("runes count", len(runes))

	sort.Slice(runes, func(i, j int) bool {
		return runes[i] < runes[j]
	})

	fmt.Println("min", int(runes[0]), string(runes[0]))
	fmt.Println("max", int(runes[len(runes)-1]), string(runes[len(runes)-1]))

	if _, err := fileOut.WriteString(string(runes)); err != nil {
		log.Fatalln(err)
	}
}
