package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"sort"
)

var fIn = "./examples/transformer-bpe/data/rus_sentences.txt"

// var fOut = "./examples/transformer-bpe/data/rus_sentences.alphabet"
var fOut = "./examples/transformer-bpe/cmd/bpe-rusent-alphabet/test"

var special = map[token]struct{}{
	" ":  {},
	"\n": {},
	".":  {},
	",":  {},
	"!":  {},
	"?":  {},
	"\"": {},
	"'":  {},
}

type token string

type tokenInfo struct {
	token token
	count int
}

func readFromFile() (context []token, err error) {
	fileIn, err := os.OpenFile(fIn, os.O_RDONLY, os.ModePerm)
	if err != nil {
		log.Fatalln(err)
	}
	defer fileIn.Close()

	b, err := io.ReadAll(fileIn)
	if err != nil {
		return nil, err
	}

	for _, r := range []rune(string(b)) {
		context = append(context, token(r))
	}
	return
}

func extractPairs(context []token, threshold float64) (tokens []tokenInfo) {
	counter := map[token]int{}
	for i := 0; i < len(context)-1; i += 2 {
		if _, ok := special[context[i]]; ok {
			i--
			continue
		}
		if _, ok := special[context[i+1]]; ok {
			i--
			continue
		}
		counter[context[i]+context[i+1]]++
	}

	max := 0
	for _, i := range counter {
		if max == 0 || i > max {
			max = i
		}
	}

	for t, i := range counter {
		if i < int(threshold*float64(max)) {
			continue
		}

		tokens = append(tokens, tokenInfo{
			token: t,
			count: i,
		})
	}

	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i].count > tokens[j].count
	})
	return
}

func updateContext(context []token, tokens []tokenInfo) (newContext []token) {
	for i := 0; i < len(context)-1; i += 2 {
		t := context[i] + context[i+1]
		m := false

		for _, info := range tokens {
			if m = t == info.token; m {
				newContext = append(newContext, t)
				break
			}
		}

		if !m {
			newContext = append(newContext, context[i])
			i--
		}
	}
	return
}

func extractTokens(context []token) (tokens []token) {
	seen := map[token]struct{}{}
	for _, t := range context {
		if _, ok := seen[t]; !ok {
			tokens = append(tokens, t)
			seen[t] = struct{}{}
		}
	}
	sort.Slice(tokens, func(i, j int) bool {
		if len(tokens[i]) > len(tokens[j]) {
			return true
		}

		if len(tokens[i]) == len(tokens[j]) {
			return tokens[i] > tokens[j]
		}

		return false
	})
	return
}

func main() {
	context, err := readFromFile()
	if err != nil {
		log.Fatalln(err)
	}

	fmt.Println("extract pairs")
	tokens := extractPairs(context, 0.1)
	context = updateContext(context, tokens)

	fmt.Println("extract pairs")
	tokens = extractPairs(context, 0.1)
	context = updateContext(context, tokens)

	fmt.Println("extract pairs")
	tokens = extractPairs(context, 0.1)
	context = updateContext(context, tokens)

	fmt.Println("extract alphabet")
	alphabet := extractTokens(context)

	fmt.Println(context[:100])
	fmt.Println(tokens)
	//for i, t := range alphabet {
	//	fmt.Println("token", i, t)
	//}
	//fmt.Println(alphabet)
	fmt.Println("alphabet length", len(alphabet))
}
