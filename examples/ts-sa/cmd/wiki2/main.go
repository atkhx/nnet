package main

import (
	"log"
	"os"

	"github.com/m-m-f/gowiki"
	_ "github.com/m-m-f/gowiki"
)

var fn = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki-20230701-pages-articles-multistream1.xml-p1p224167"

func main() {
	f, err := os.OpenFile(fn, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	gowiki.PageGetter()
}
