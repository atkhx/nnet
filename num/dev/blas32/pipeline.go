package blas32

import (
	"context"
	"runtime"
	"sync"

	"github.com/atkhx/nnet/num"
)

func NewPipeline(lastNode *num.Data) (out *Pipeline) {
	out = &Pipeline{
		wg:  &sync.WaitGroup{},
		ctx: context.Background(),

		forwardChan:    make(chan func(ctx context.Context)),
		forwardLayers:  getForwardNodeLayers(lastNode),
		backwardLayers: getBackwardNodeLayers(lastNode),
		resetLayers:    getResetGradsNodeLayers(lastNode),
		resetGradsFunc: getResetGradsNodeFuncs(lastNode),
	}
	//out.runRoutines()
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
}

func (p *Pipeline) runRoutines() {
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
		//p.wg.Add(len(nodes))
		for _, node := range nodes {
			//p.forwardChan <- node.calcData
			node.CalcData(ctx)
		}
		//p.wg.Wait()
	}
}

func (p *Pipeline) Backward(ctx context.Context) {
	for _, resetFunc := range p.resetGradsFunc {
		resetFunc(p.ctx)
	}

	for _, nodes := range p.backwardLayers {
		//p.wg.Add(len(nodes))
		for _, node := range nodes {
			//p.forwardChan <- node.calcGrad
			node.CalcGrad(ctx)
			//p.forwardChan <- node.calcGrad
		}
		//p.wg.Wait()
	}
}