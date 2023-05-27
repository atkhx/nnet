package layer

import (
	"sync"

	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
	initWeights initializer.Initializer,
) *SAMultiHead {
	return &SAMultiHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		headsCount:    headsCount,
		initWeights:   initWeights,
	}
}

type SAHeadParams struct {
	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data
	outputObj  *num.Data
}

type SAMultiHead struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int
	headsCount    int

	Heads []SAHeadParams

	concatObj *num.Data
	forUpdate num.Nodes

	wg sync.WaitGroup
}

func (l *SAMultiHead) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.Heads = make([]SAHeadParams, l.headsCount)

	outputObjs := make([]*num.Data, 0, l.headsCount)

	for i := 0; i < l.headsCount; i++ {
		l.Heads[i].KeyWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
		l.Heads[i].QryWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
		l.Heads[i].ValWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

		l.Heads[i].outputObj = inputs.SAHead(
			l.headSize,
			l.Heads[i].KeyWeights,
			l.Heads[i].QryWeights,
			l.Heads[i].ValWeights,
		)

		outputObjs = append(outputObjs, l.Heads[i].outputObj)

		l.forUpdate = append(l.forUpdate,
			l.Heads[i].KeyWeights,
			l.Heads[i].QryWeights,
			l.Heads[i].ValWeights,
		)
	}

	l.concatObj = outputObjs[0].ConcatRows(outputObjs[1:]...)

	return l.concatObj
}

func (l *SAMultiHead) Forward() {
	l.wg.Add(l.headsCount)
	for i := range l.Heads {
		go func(i int) {
			l.Heads[i].outputObj.Forward()
			l.wg.Done()
		}(i)
	}
	l.wg.Wait()

	l.concatObj.Forward()
}

func (l *SAMultiHead) Backward() {
	l.concatObj.Backward()

	l.wg.Add(l.headsCount)
	for i := range l.Heads {
		go func(i int) {
			l.Heads[i].outputObj.Backward()
			l.wg.Done()
		}(i)
	}
	l.wg.Wait()
}

func (l *SAMultiHead) ForUpdate() num.Nodes {
	return l.forUpdate
}
