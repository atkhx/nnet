package metal

import (
	"context"
	"sync"

	"github.com/atkhx/mps"
	"github.com/atkhx/nnet/num"
)

func NewPipeline(lastNode *num.Data) (out *Pipeline) {
	out = &Pipeline{
		wg:  &sync.WaitGroup{},
		ctx: context.Background(),

		forwardLayers:  getForwardNodeLayers(lastNode),
		backwardLayers: getBackwardNodeLayers(lastNode),
		resetGradsFunc: getResetGradsNodeFuncs(lastNode),
		commandQueue:   mps.DefaultDevice.CreateCommandQueue(),
	}
	return
}

type Pipeline struct {
	wg  *sync.WaitGroup
	ctx context.Context

	forwardLayers  NodeLayers
	backwardLayers NodeLayers
	resetGradsFunc []func(ctx context.Context)
	commandQueue   *mps.MTLCommandQueue
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

func (p *Pipeline) Backward(ctx context.Context) {
	for _, resetFunc := range p.resetGradsFunc {
		resetFunc(p.ctx)
	}

	for _, nodes := range p.backwardLayers {
		commandBuffer := p.commandQueue.GetCommandBuffer()

		ctx := mps.ContextWithCommandBuffer(ctx, commandBuffer)
		for _, node := range nodes {
			node.CalcGrad(ctx)
		}
		commandBuffer.Wait()
	}
}
