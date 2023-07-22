package num

import (
	"runtime"
	"sync"
)

func NewPipeline(lastNode *Data) (out *Pipeline) {
	forwardLayers := getForwardNodeLayers(lastNode)
	backwardLayers := getBackwardNodeLayers(lastNode)
	resetLayers := getResetGradsNodeLayers(lastNode)

	parallel := runtime.GOMAXPROCS(0)

	wg := &sync.WaitGroup{}

	fChan := make(chan func())

	for i := 0; i < parallel; i++ {
		go func() {
			for fn := range fChan {
				fn()
				wg.Done()
			}
		}()
	}

	return &Pipeline{
		wg: wg,

		forwardChan:   fChan,
		forwardLayers: forwardLayers,

		//backwardChan:   bChan,
		backwardLayers: backwardLayers,

		//resetChan:   rChan,
		resetLayers: resetLayers,
	}
}

type Pipeline struct {
	wg *sync.WaitGroup

	forwardChan   chan func()
	forwardLayers NodeLayers

	//backwardChan   chan *Data
	backwardLayers NodeLayers

	//resetChan   chan *Data
	resetLayers NodeLayers
}

func (p *Pipeline) Forward() {
	for _, nodes := range p.forwardLayers {
		if len(nodes) == 1 {
			nodes[0].calcData()
			continue
		}

		p.wg.Add(len(nodes))
		for _, node := range nodes {
			p.forwardChan <- node.calcData
		}
		p.wg.Wait()
	}
}

func (p *Pipeline) Backward() {
	//p.resetLayers[0][0].Grad.Ones()
	//for _, nodes := range p.resetLayers[1:] {
	for i, nodes := range p.resetLayers {
		//	if len(nodes) == 1 {
		//		nodes[0].Grad.Zero()
		//		continue
		//	}

		p.wg.Add(len(nodes))
		for _, node := range nodes {
			if i == 0 {
				p.forwardChan <- node.Grad.Ones
			} else {
				p.forwardChan <- node.Grad.Zero
			}
			//p.forwardChan <- node.Grad.Zero
		}
	}
	p.wg.Wait()

	for _, nodes := range p.backwardLayers {
		if len(nodes) == 1 {
			nodes[0].calcGrad()
			continue
		}

		p.wg.Add(len(nodes))
		for _, node := range nodes {
			p.forwardChan <- node.calcGrad
		}
		p.wg.Wait()
	}
}
