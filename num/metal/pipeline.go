package metal

import (
	"context"
	"runtime"
	"sync"

	"github.com/atkhx/mps"
)

func NewPipeline(lastNode *Data) (out *Pipeline) {
	out = &Pipeline{
		wg:  &sync.WaitGroup{},
		ctx: context.Background(),

		forwardChan:    make(chan func(ctx context.Context)),
		forwardLayers:  getForwardNodeLayers(lastNode),
		backwardLayers: getBackwardNodeLayers(lastNode),
		resetLayers:    getResetGradsNodeLayers(lastNode),
		resetGradsFunc: getResetGradsNodeFuncs(lastNode),
	}
	out.runRoutines()
	return
}

type Pipeline struct {
	wg  *sync.WaitGroup
	ctx context.Context

	forwardChan    chan func(ctx context.Context)
	forwardLayers  NodeLayers
	backwardLayers NodeLayers
	resetLayers    NodeLayers
	resetGradsFunc []func(ctx context.Context)
	commandQueue   *mps.MTLCommandQueue
}

func (p *Pipeline) runRoutines() {
	p.commandQueue = mps.DefaultDevice.CreateCommandQueue()

	parallel := runtime.GOMAXPROCS(0)
	for i := 0; i < parallel; i++ {
		go func() {
			for fn := range p.forwardChan {
				fn(p.ctx)
				p.wg.Done()
			}
		}()
	}
}

func (p *Pipeline) Forward(ctx context.Context) {
	for _, nodes := range p.forwardLayers {
		commandBuffer := p.commandQueue.GetCommandBuffer()
		p.ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
		p.wg.Add(len(nodes))
		for _, node := range nodes {
			p.forwardChan <- node.calcData
		}
		p.wg.Wait()
		commandBuffer.Wait()
	}
}

func (p *Pipeline) Backward(ctx context.Context) {
	{
		commandBuffer := p.commandQueue.GetCommandBuffer()
		p.ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
		for _, resetFunc := range p.resetGradsFunc {
			resetFunc(p.ctx)
		}
		commandBuffer.Wait()
	}

	for _, nodes := range p.backwardLayers {
		commandBuffer := p.commandQueue.GetCommandBuffer()

		p.ctx = mps.ContextWithCommandBuffer(ctx, commandBuffer)
		p.wg.Add(len(nodes))
		for _, node := range nodes {
			p.forwardChan <- node.calcGrad
		}
		p.wg.Wait()
		commandBuffer.Wait()
		p.ctx = ctx
	}
}
