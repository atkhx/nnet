package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
)

var fn = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki1.txt"
var of = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki1.alphabet"

func main() {
	f, err := os.OpenFile(fn, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	out, err := os.OpenFile(of, os.O_CREATE|os.O_RDWR, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer out.Close()

	reader := bufio.NewReader(f)

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

	if _, err := out.WriteString(string(runes)); err != nil {
		log.Fatalln(err)
	}
}
