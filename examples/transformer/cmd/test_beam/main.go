package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"

	"github.com/atkhx/nnet/examples/transformer/dataset"
	"github.com/atkhx/nnet/examples/transformer/pkg"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/transformer/config.json", "nn config file")
	flag.Parse()
}

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	batchSize := 1

	trainDataset := dataset.NewDataset(pkg.ContextLength, batchSize)
	trainDataset.ParseAlphabet()

	model := pkg.CreateNN(
		trainDataset.GetAlphabetSize(),
		batchSize,
		nil,
	)

	output := model.Compile()
	if err := model.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := num.NewPipeline(output)

	inputIndexes := trainDataset.EncodeString(``)
	inputTokens := trainDataset.Decode(inputIndexes...)

	fmt.Print(trainDataset.Decode(inputIndexes...))

	getNextProbs := getNextProbsFunc(
		trainDataset,
		model,
		pipeline,
		output,
	)

	fmt.Println()
	fmt.Println("---")
	//
	//for {
	//	l := len(inputTokens)
	//	t := beamSearch(getNextProbs, inputTokens, 5, 32)
	//	//inputTokens = append(inputTokens, t...)
	//	//inputTokens = t[l:]
	//	inputTokens = t
	//	fmt.Print(t[l:])
	//}

	result := beamSearch(getNextProbs, inputTokens, 2, 128)
	fmt.Println(strings.Repeat("-", 40))
	fmt.Println(result)
	fmt.Println()
}

func getNextProbsFunc(
	trainDataset *dataset.Dataset,
	model *model.Sequential,
	pipeline *num.Pipeline,
	output *num.Data,
) func(inputBytes dataset.Tokens) (num.Float64s, []dataset.Token) {
	return func(originalInputTokens dataset.Tokens) (num.Float64s, []dataset.Token) {
		inputTokens := originalInputTokens.Copy()

		if len(inputTokens) > pkg.ContextLength {
			inputTokens = inputTokens[len(inputTokens)-pkg.ContextLength:]
		}

		inputsFloat := trainDataset.EncodeToFloats(inputTokens...)

		copy(model.GetInput().Data, inputsFloat)
		pipeline.Forward()

		pos := len(inputTokens) - 1
		if pos < 0 {
			pos = 0
		}

		outputData := output.Data[pos*trainDataset.GetAlphabetSize() : (pos+1)*trainDataset.GetAlphabetSize()]
		outputData.Softmax()

		probs := outputData.Copy()
		tokens := []dataset.Token{}

		for i := range probs {
			tokens = append(tokens, trainDataset.Decode(i)[0])
		}

		return probs, tokens
	}
}

type BeamSearchNode struct {
	Sequence dataset.Tokens // Последовательность токенов
	Score    float64        // Оценка для последовательности
	Scores   num.Float64s
}

func beamSearch(
	nextProbsFunc func(inputBytes dataset.Tokens) (num.Float64s, []dataset.Token),
	input dataset.Tokens,
	beamSize int,
	maxSteps int,
) dataset.Tokens {
	// Инициализация луча
	beams := []*BeamSearchNode{
		{
			Sequence: input,
			Score:    0.0,
			Scores:   num.Float64s{0},
		},
	}

	// Пошаговый процесс beams search
	for step := 0; step < maxSteps; step++ {
		var candidates []*BeamSearchNode

		// Расширение текущих последовательностей
		for _, node := range beams {
			// Получение распределения вероятностей для следующего токена от модели
			probs, tokens := nextProbsFunc(node.Sequence)

			////Сэмплирование следующих токенов
			//for i, prob := range probs {
			//	seq := make(dataset.Tokens, len(node.Sequence))
			//	copy(seq, node.Sequence)
			//	seq = append(seq, tokens[i])
			//
			//	candidates = append(candidates, &BeamSearchNode{
			//		Sequence: seq,
			//		Score:    0.5*node.Score + math.Log(prob),
			//	})
			//}

			ppp := probs.Copy()
			ppp.CumulativeSum()

			usedTokens := map[dataset.Token]struct{}{}
			for j := 0; j < beamSize; j++ {
				i := ppp.Multinomial()
				t := tokens[i]

				if _, ok := usedTokens[t]; ok {
					j--
					continue
				}

				usedTokens[t] = struct{}{}

				seq := make(dataset.Tokens, len(node.Sequence))
				copy(seq, node.Sequence)
				seq = append(seq, t)

				//scoresSum := node.Scores.Sum()
				//score := scoresSum/float64(len(seq)) + math.Log(probs[i])
				//score := math.Log(0.5*node.Score + probs[i])
				//score := 0.5*node.Score + math.Log(probs[i])
				//score := node.Score + math.Log(probs[i])
				score := (node.Score + math.Log(probs[i])) / 2.
				//scores := node.Scores.Copy()
				//scores = append(scores, score)

				candidates = append(candidates, &BeamSearchNode{
					Sequence: seq,
					//Score:    0.5*node.Score + math.Log(probs[i]),
					Score: score,
					//Scores: scores,
					//Score: node.Score * probs[i],
					//Score: math.Log(node.Score + probs[i]),
				})
			}

		}

		// Сортировка кандидатов по оценке
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Score > candidates[j].Score
		})

		//for _, candidate := range candidates[:beamSize] {
		//	fmt.Println(candidate.Sequence, "score", candidate.Score)
		//}
		//
		for i := beamSize; i > 0; i-- {
			candidate := candidates[i-1]
			fmt.Println(candidate.Sequence, "score", candidate.Score)
		}

		//fmt.Println("step", step)
		//if step > 10 {
		//	os.Exit(0)
		//}
		// Выбор лучших k кандидатов
		beams = candidates[:beamSize]
	}

	//result := []dataset.Tokens{}
	//for _, beam := range beams {
	//	//fmt.Println("beam.Score", beam.Score)
	//	result = append(result, beam.Sequence)
	//}
	//return result
	//
	//// Возвращение последовательности с наивысшей оценкой
	////bestSequence := beams[0].Sequence
	return beams[0].Sequence
}
