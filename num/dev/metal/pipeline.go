package metal

import (
	"context"

	"github.com/atkhx/mps"
	"github.com/atkhx/nnet/num"
)

func NewPipeline(device *Device, lastNode *num.Data) *Pipeline {
	return &Pipeline{
		device:          device,
		forwardLayers:   num.GetForwardNodeLayers(lastNode),
		backwardLayers:  num.GetBackwardNodeLayers(lastNode),
		resetGradsNodes: num.GetResetGradsNodes(lastNode),
		commandQueue:    device.CreateCommandQueue(),
	}
}

type Pipeline struct {
	device *Device

	forwardLayers   num.NodeLayers
	backwardLayers  num.NodeLayers
	resetGradsNodes num.Nodes
	commandQueue    *mps.MTLCommandQueue
}

func (p *Pipeline) Forward(ctx context.Context) {
	for _, nodes := range p.forwardLayers {
		commandBuffer := p.commandQueue.GetCommandBuffer()
		ctx := mps.ContextWithCommandBuffer(ctx, commandBuffer)
		for _, node := range nodes {
			node.CalcData(ctx)
		}
		commandBuffer.Wait()
	}
}

func (p *Pipeline) resetGrads(ctx context.Context) {
	commandBuffer := p.commandQueue.GetCommandBuffer()
	for i, node := range p.resetGradsNodes {
		if i == 0 {
			commandBuffer.FillMTLBuffer(node.Opts.(dataOpts).gradBuffer, 1.0)
		} else {
			commandBuffer.ClearMTLBuffer(node.Opts.(dataOpts).gradBuffer)
		}
	}
	commandBuffer.Wait()
}

func (p *Pipeline) calcGrads(ctx context.Context) {
	for _, nodes := range p.backwardLayers {
		commandBuffer := p.commandQueue.GetCommandBuffer()
		ctx := mps.ContextWithCommandBuffer(ctx, commandBuffer)
		for _, node := range nodes {
			node.CalcGrad(ctx)
		}
		commandBuffer.Wait()
	}
}

func (p *Pipeline) Backward(ctx context.Context) {
	p.resetGrads(ctx)
	p.calcGrads(ctx)
}
