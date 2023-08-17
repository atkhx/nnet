package main

import (
	"bufio"
	"log"
	"os"
	"regexp"
)

var fIn = "./examples/transformer/src/rus_sentences.tsv"
var fOut = "./examples/transformer/data/rus_sentences.txt"

func main() {
	os.Remove(fOut)

	fileIn, err := os.OpenFile(fIn, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer fileIn.Close()

	fileOut, err := os.OpenFile(fOut, os.O_CREATE|os.O_RDWR|os.O_APPEND, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer fileOut.Close()

	re := regexp.MustCompile(`^(\d+\trus\t)`)

	i := 0
	reader := bufio.NewReader(fileIn)
	for {
		i++
		r, _, err := reader.ReadLine()
		if err != nil {
			log.Println(err)
			break
		}

		r = re.ReplaceAll(r, []byte{})
		r = append(r, []byte("\n")...)
		fileOut.Write(r)
	}
}
