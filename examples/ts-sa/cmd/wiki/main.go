package main

import (
	"fmt"
	"log"
	"os"
	"regexp"

	"github.com/dustin/go-wikiparse"
	"github.com/m-m-f/gowiki"
)

var fn = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki-20230701-pages-articles-multistream1.xml-p1p224167"
var outfile = "/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet/examples/ts-sa/dataset/ruwiki1.txt"

var repl = regexp.MustCompile(`[­]+`)
var nbsp = regexp.MustCompile(`[  ]+`)
var white = regexp.MustCompile(`[ \t]+`)
var nl = regexp.MustCompile(`[\n]+`)
var brakets = regexp.MustCompile(`\(\s*\)`)
var sqbr = regexp.MustCompile(`\[\[.*?\]\]`)
var sqbr2 = regexp.MustCompile(`\[.*?\]`)
var nlspace = regexp.MustCompile(`\n[\s]+`)
var nlfspace = regexp.MustCompile(`[\s+]\n`)

func main() {
	f, err := os.OpenFile(fn, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	out, err := os.OpenFile(outfile, os.O_CREATE|os.O_RDWR|os.O_APPEND, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer out.Close()

	p, err := wikiparse.NewParser(f)
	if err != nil {
		log.Fatalln("Error setting up parser", err)
	}

	i := 0
	for err == nil {

		var page *wikiparse.Page
		page, err = p.Next()
		if err == nil {
			i++
			if i == 1 {
				continue
			}

			if i%1000 == 0 {
				fmt.Println("i", i, page.Title)
			}

			a, err := gowiki.ParseArticle(page.Title, page.Revisions[0].Text, &gowiki.DummyPageGetter{})
			if err != nil {
				log.Println(err)
				break
			}

			if _, err := writeToOut(out, page.Title); err != nil {
				log.Println(err)
				break
			}

			if _, err := writeToOut(out, a.GetAbstract()); err != nil {
				log.Println(err)
				break
			}

			if _, err := writeToOut(out, a.GetText()); err != nil {
				log.Println(err)
				break
			}
		}
	}
}

func writeToOut(out *os.File, text string) (int, error) {
	text = repl.ReplaceAllString(text, "")
	text = nbsp.ReplaceAllString(text, " ")
	text = white.ReplaceAllString(text, " ")
	text = nl.ReplaceAllString(text, "\n")
	text = brakets.ReplaceAllString(text, "")
	text = sqbr.ReplaceAllString(text, "")
	text = sqbr2.ReplaceAllString(text, "")
	text = nlspace.ReplaceAllString(text, "\n")
	text = nlfspace.ReplaceAllString(text, "\n")

	return out.WriteString(text)
}
