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

func (p *Pipeline) Reset() {
	commandBuffer := p.commandQueue.GetCommandBuffer()
	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			commandBuffer.ClearMTLBuffer(node.Data)
			commandBuffer.ClearMTLBuffer(node.Grad)
		}
	}
	commandBuffer.Wait()
}

func (p *Pipeline) Forward(ctx context.Context) {
	commandBuffer := p.commandQueue.GetCommandBuffer()
	ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			node.CalcData(ctx)
		}
	}
	commandBuffer.Wait()
}

// func (p *Pipeline) resetGrads(commandBuffer *mps.MTLCommandBuffer) {
func (p *Pipeline) resetGrads() {
	commandBuffer := p.commandQueue.GetCommandBuffer()
	for i, node := range p.resetGradsNodes {
		if i == 0 {
			commandBuffer.FillMTLBuffer(node.Grad, 1.0)
		} else {
			commandBuffer.ClearMTLBuffer(node.Grad)
		}
	}
	commandBuffer.Wait()
}

func (p *Pipeline) calcGrads(ctx context.Context) {
	commandBuffer := p.commandQueue.GetCommandBuffer()
	ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
	for _, nodes := range p.backwardLayers {
		for _, node := range nodes {
			node.CalcGrad(ctx)
		}
	}
	commandBuffer.Wait()
}

func (p *Pipeline) Backward(ctx context.Context) {
	//commandBuffer := p.commandQueue.GetCommandBuffer()
	//ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
	//p.resetGrads(commandBuffer)
	p.resetGrads()
	p.calcGrads(ctx)
	//commandBuffer.Wait()
}

func (p *Pipeline) TrainIteration(ctx context.Context, update func(ctx context.Context)) {
	commandBuffer := p.commandQueue.GetCommandBuffer()
	ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)

	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			node.CalcData(ctx)
		}
	}

	for i, node := range p.resetGradsNodes {
		if i == 0 {
			commandBuffer.FillMTLBuffer(node.Grad, 1.0)
		} else {
			commandBuffer.ClearMTLBuffer(node.Grad)
		}
	}

	for _, nodes := range p.backwardLayers {
		for _, node := range nodes {
			node.CalcGrad(ctx)
		}
	}

	update(ctx)

	commandBuffer.Wait()
	commandBuffer.Release()
}

func (p *Pipeline) GetCommandBuffer() *mps.MTLCommandBuffer {
	return p.commandQueue.GetCommandBuffer()
}
