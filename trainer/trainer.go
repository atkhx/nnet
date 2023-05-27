//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package trainer

import (
	"github.com/atkhx/nnet/num"
)

func New(update num.Nodes, opts ...Option) *Trainer {
	res := &Trainer{update: update}
	applyOptions(res, defaults...)
	applyOptions(res, opts...)
	res.method.Init(res.getParamsCount())
	return res
}

type Trainer struct {
	l1Decay float64
	l2Decay float64

	update num.Nodes
	method Method
}

func (t *Trainer) getParamsCount() (weightsCount int) {
	for _, node := range t.update {
		weightsCount += len(node.Data)
	}
	return
}

func (t *Trainer) UpdateWeights() {
	offset := 0

	for _, node := range t.update {
		for j := 0; j < len(node.Data); j++ {
			l1grad := t.l1Decay
			if node.Data[j] <= 0 {
				l1grad = -l1grad
			}

			l2grad := t.l2Decay * node.Data[j]
			gradient := l2grad + l1grad + node.Grad[j]

			node.Data[j] += t.method.GetDelta(offset+j, gradient)
		}
		offset += len(node.Data)
	}
}
