package native

import (
	"context"

	"github.com/atkhx/nnet/num"
)

func NewPipeline(device *Device, lastNode *num.Data) *Pipeline {
	return &Pipeline{
		device: device,

		forwardLayers:   num.GetForwardNodeLayers(lastNode),
		backwardLayers:  num.GetBackwardNodeLayers(lastNode),
		resetGradsNodes: num.GetResetGradsNodes(lastNode),
	}
}

type Pipeline struct {
	device *Device

	forwardLayers   num.NodeLayers
	backwardLayers  num.NodeLayers
	resetGradsNodes num.Nodes
}

func (p *Pipeline) Forward(ctx context.Context) {
	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			node.CalcData(ctx)
		}
	}
}

func (p *Pipeline) Backward(ctx context.Context) {
	for i, node := range p.resetGradsNodes {
		if i == 0 {
			p.device.FillGradWithOnes(node)
		} else {
			p.device.FillGradWithZeros(node)
		}
	}

	for _, nodes := range p.backwardLayers {
		for _, node := range nodes {
			node.CalcGrad(ctx)
		}
	}
}
