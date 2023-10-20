package main

import (
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/dustin/go-wikiparse"
	"github.com/m-m-f/gowiki"
)

var fIn = "./gpt/src/ruwiki-20230701-pages-articles-multistream1.xml-p1p224167"
var fOut = "./gpt/data/ruwiki12.txt"

var replaceSymbols = regexp.MustCompile(`[­]+`)
var replaceFiles = regexp.MustCompile(`\[\[[^\]]+\]\]`)
var replaceLinks = regexp.MustCompile(`\[.*?\]`)
var replaceCategories = regexp.MustCompile(`:[^:]+:`)

var whiteSpaces = regexp.MustCompile(`[ \t]+`)
var replaceNBSPWithSpaces = regexp.MustCompile(`[  ]+`)

var newLines = regexp.MustCompile(`\n{2,}`)
var brakets = regexp.MustCompile(`\(\s*\)`)
var nlspace = regexp.MustCompile(`\n[\s]+`)
var nlfspace = regexp.MustCompile(`[\s+]\n`)

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

	p, err := wikiparse.NewParser(fileIn)
	if err != nil {
		log.Fatalln("Error setting up parser", err)
	}

	i := 0
	for err == nil {
		page, err := p.Next()
		if err != nil {
			log.Println("err", err)
			break
		}
		i++
		if i == 1 {
			continue
		}

		if strings.Contains(page.Title, "язык") {
			fmt.Println("skip", page.Title)
			continue
		}

		//if i == 3 {
		//	break
		//}

		if i%1000 == 0 {
			fmt.Println("i", i, page.Title)
		}

		a, err := gowiki.ParseArticle(page.Title, page.Revisions[0].Text, &gowiki.DummyPageGetter{})
		if err != nil {
			log.Println(err)
			break
		}

		//if _, err := writeToOut(fileOut, a.Title); err != nil {
		//	log.Println(err)
		//	break
		//}
		//
		//if _, err := writeToOut(fileOut, a.GetAbstract()); err != nil {
		//	log.Println(err)
		//	break
		//}
		//if _, err := writeToOut(fileOut, a.GetText()); err != nil {
		//	log.Println(err)
		//	break
		//}

		if _, err := writeToOut(fileOut, a.GetText()); err != nil {
			log.Println(err)
			break
		}
	}
}

func writeToOut(out *os.File, text string) (int, error) {
	text = replaceSymbols.ReplaceAllString(text, "")
	text = replaceNBSPWithSpaces.ReplaceAllString(text, " ")
	text = replaceFiles.ReplaceAllString(text, "")
	text = replaceLinks.ReplaceAllString(text, "")
	text = replaceCategories.ReplaceAllString(text, " ")
	text = newLines.ReplaceAllString(text, "\n")

	text = brakets.ReplaceAllString(text, "")
	text = nlspace.ReplaceAllString(text, "\n")
	text = nlfspace.ReplaceAllString(text, "\n")

	text = whiteSpaces.ReplaceAllString(text, " ")
	return out.WriteString(text)
}
