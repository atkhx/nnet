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
	namesDataset.ParseAlphabet(dataset.RuWiki1)

	seqModel := pkg.CreateNN(
		namesDataset.GetAlphabetSize(),
		batchSize,
	)

	output := seqModel.Compile()
	if err := seqModel.LoadFromFile(filename); err != nil {
		log.Fatalln(err)
	}

	pipeline := num.NewPipeline(output)
	inputsFloat := make(num.Float64s, pkg.ContextLength)

	//batchInputs, _ := namesDataset.ReadRandomSampleBatch()
	//copy(inputsFloat, batchInputs)

	for j := 0; j < 10000; j++ {
		//inputsFloat := namesDataset.EncodeToFloats(inputTokens...)

		copy(seqModel.GetInput().Data, inputsFloat)
		pipeline.Forward()

		output := output.Data[(pkg.ContextLength-1)*namesDataset.GetAlphabetSize():]

		//output.Softmax()
		//f, _ := output.MaxKeyVal()
		//output.CumulativeSum()
		//f := output.Multinomial()

		f := sampleWithTemperature(output, 1)
		b := namesDataset.Decode(f)[0]

		inputsFloat = append(inputsFloat, float64(f))
		inputsFloat = inputsFloat[1:]

		//for _, token := range b {
		fmt.Print(string(b))
		//}
	}

	//for _, t := range namesDataset.DecodeFloats(inputsFloat...) {
	//	fmt.Print(t)
	//}
	//
	//fmt.Println()
	//fmt.Println("---")
	//
	//res := beamSearch(func(ints []int) []float64 {
	//	floats := make([]float64, len(ints))
	//	for i, v := range ints {
	//		floats[i] = float64(v)
	//	}
	//
	//	inp := make([]float64, pkg.ContextLength)
	//	//copy(inp, floats)
	//	inp = append(inp, floats...)
	//	inp = inp[len(inp)-pkg.ContextLength:]
	//
	//	//seqModel.GetInput().Data = inp
	//	seqModel.GetInput().Data.Zero()
	//	copy(seqModel.GetInput().Data, inp)
	//	//fmt.Println(seqModel.GetInput().Data)
	//	pipeline.Forward()
	//
	//	return output.Data[(pkg.ContextLength-1)*namesDataset.GetAlphabetSize():].NewLinkedCopy()
	//}, inputsFloat.ToInt(), 1, 10)
	//
	//for _, t := range namesDataset.Decode(res...) {
	//	fmt.Print(t)
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

func beamSearch(predict func([]int) []float64, input []int, beamSize int, maxSteps int) []int {
	// Инициализация луча
	beam := make([]*BeamSearchNode, 1)
	beam[0] = &BeamSearchNode{
		Sequence: input,
		Score:    0.0,
	}

	// Пошаговый процесс beam search
	for step := 0; step < maxSteps; step++ {
		candidates := make([]*BeamSearchNode, 0)

		// Расширение текущих последовательностей
		for _, node := range beam {
			// Получение распределения вероятностей для следующего токена от модели
			logits := predict(node.Sequence)
			probs := softmax(logits)

			// Сэмплирование следующих токенов
			for token, prob := range probs {
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

		// Выбор лучших k кандидатов
		beam = candidates[:beamSize]
	}

	// Возвращение последовательности с наивысшей оценкой
	bestSequence := beam[0].Sequence

	return bestSequence
}
