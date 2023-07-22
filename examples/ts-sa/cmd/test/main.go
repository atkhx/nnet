package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"

	"github.com/atkhx/nnet/examples/ts-sa/dataset"
	"github.com/atkhx/nnet/examples/ts-sa/pkg"
	"github.com/atkhx/nnet/num"
)

var filename string

func init() {
	flag.StringVar(&filename, "c", "./examples/ts-sa/config.json", "nn config file")
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

	namesDataset := dataset.NewDataset(pkg.ContextLength, batchSize)
	namesDataset.ParseAlphabet()

	seqModel := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		batchSize,
	)

	output := seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := num.NewPipeline(output)

	inp := namesDataset.EncodeString(`Привет! `)
	inputBytes := namesDataset.Decode(inp...)

	fmt.Print(namesDataset.Decode(inp...))

	for j := 0; j < 10000; j++ {
		inputsFloat := namesDataset.EncodeToFloats(inputBytes...)

		copy(seqModel.GetInput().Data, inputsFloat)
		pipeline.Forward()

		pos := len(inputBytes) - 1
		if pos < 0 {
			pos = 0
		}

		output := output.Data[pos*namesDataset.GetAlphabetSize() : (pos+1)*namesDataset.GetAlphabetSize()]
		//output.Softmax()
		//output.CumulativeSum()
		//f := output.Multinomial()

		f := sampleWithTemperature(output, 0.7)
		b := namesDataset.Decode(f)

		inputBytes = append(inputBytes, b...)
		if len(inputBytes) > pkg.ContextLength {
			l := len(inputBytes)
			inputBytes = inputBytes[l-pkg.ContextLength:]
		}

		fmt.Print(b)
	}

	//fmt.Println()
	//fmt.Println("---")
	//
	//results := beamSearch(func(ints []int) []float64 {
	//	floats := make([]float64, len(ints))
	//	for i, v := range ints {
	//		floats[i] = float64(v)
	//	}
	//
	//	if len(floats) > pkg.ContextLength {
	//		floats = floats[len(floats)-pkg.ContextLength:]
	//	}
	//
	//	pos := len(floats) - 1
	//	if pos < 0 {
	//		pos = 0
	//	}
	//
	//	seqModel.GetInput().Data.Zero()
	//	copy(seqModel.GetInput().Data, floats)
	//	pipeline.Forward()
	//
	//	return output.Data[pos*namesDataset.GetAlphabetSize() : (pos+1)*namesDataset.GetAlphabetSize()].Copy()
	//}, namesDataset.Encode(inputBytes...), 10, 50)
	//
	//for _, result := range results {
	//	fmt.Println(namesDataset.Decode(result...))
	//	fmt.Println()
	//}

	fmt.Println()
}

func sampleWithTemperature(logits num.Float64s, temperature float64) int {
	// Применяем температуру к логитам
	for i := 0; i < len(logits); i++ {
		logits[i] /= temperature
	}

	// Преобразуем логиты в вероятности с помощью Softmax
	probs := softmax(logits)
	//probs := logits.NewLinkedCopy()
	//probs.Softmax()

	// Сэмплируем следующий токен с учетом вероятностей
	sampledIndex := sampleFromDistribution(probs)

	return sampledIndex
}

func softmax(logits []float64) []float64 {
	probs := make([]float64, len(logits))
	sum := 0.0

	for _, logit := range logits {
		sum += math.Exp(logit)
	}

	for i, logit := range logits {
		probs[i] = math.Exp(logit) / sum
	}
	return probs
}

func sampleFromDistribution(probs []float64) int {
	r := rand.Float64()
	cumulativeProb := 0.0

	for i, prob := range probs {
		cumulativeProb += prob
		if r <= cumulativeProb {
			return i
		}
	}
	panic("nonoon")
	return rand.Intn(len(probs))
}

type BeamSearchNode struct {
	Sequence []int   // Последовательность токенов
	Score    float64 // Оценка для последовательности
}

func beamSearch(predict func([]int) []float64, input []int, beamSize int, maxSteps int) [][]int {
	// Инициализация луча
	beams := []*BeamSearchNode{
		{
			Sequence: input,
			Score:    0.0,
		},
	}

	// Пошаговый процесс beams search
	for step := 0; step < maxSteps; step++ {
		var candidates []*BeamSearchNode

		// Расширение текущих последовательностей
		for _, node := range beams {
			// Получение распределения вероятностей для следующего токена от модели
			logits := predict(node.Sequence)
			probs := softmax(logits)

			maxProbs := make([]float64, len(probs))
			copy(maxProbs, probs)
			sort.Slice(maxProbs, func(i, j int) bool {
				return maxProbs[i] > maxProbs[j]
			})

			maxProbs = maxProbs[:10]

			minMaxProb := maxProbs[len(maxProbs)-1]

			// Сэмплирование следующих токенов
			for token, prob := range probs {
				if prob < minMaxProb {
					continue
				}

				candidate := &BeamSearchNode{
					Sequence: append(node.Sequence, token),
					Score:    node.Score + math.Log(prob),
				}
				candidates = append(candidates, candidate)
			}
		}

		// Сортировка кандидатов по оценке
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Score > candidates[j].Score
		})

		//for _, candidate := range candidates {
		//	fmt.Println(candidate.Sequence)
		//}
		// Выбор лучших k кандидатов
		beams = candidates[:beamSize]
	}

	result := [][]int{}
	for _, beam := range beams {
		result = append(result, beam.Sequence)
	}
	return result

	// Возвращение последовательности с наивысшей оценкой
	//bestSequence := beams[0].Sequence
	//return bestSequence
}
