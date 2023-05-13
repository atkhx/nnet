package layer

import (
	"fmt"
	"math"
	"strings"

	"github.com/atkhx/nnet/num"
)

func NewSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
) *SAMultiHead {
	return &SAMultiHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		headsCount:    headsCount,
	}
}

type SAHeadParams struct {
	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data
	outputObj  *num.Data
}

type SAMultiHead struct {
	featuresCount int
	headSize      int
	headsCount    int

	Heads []SAHeadParams

	concatObj *num.Data
	forUpdate num.Nodes
}

func (l *SAMultiHead) Compile(inputs *num.Data) *num.Data {
	weightK := num.LinearGain / math.Pow(float64(len(inputs.Data)), 0.5)

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

	fmt.Println(strings.Repeat("-", 40))
	fmt.Println("SAM\t", "3x", l.Heads[0].KeyWeights.Dims, "out", l.concatObj.Dims, "head count", l.headsCount)

	return l.concatObj
}

//var wg = sync.WaitGroup{}

func (l *SAMultiHead) Forward() {
	//wg.Add(l.headsCount)
	for i := range l.Heads {
		//go func(i int) {
		l.Heads[i].outputObj.Forward()
		//wg.Done()
		//}(i)
	}
	//wg.Wait()
	l.concatObj.Forward()
}

func (l *SAMultiHead) ForUpdate() num.Nodes {
	return l.forUpdate
}
