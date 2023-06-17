package num

import "sync"

func NewPipeline(lastNode *Data) (out *Pipeline) {
	forwardLayers := getForwardNodeLayers(lastNode)
	backwardLayers := getBackwardNodeLayers(lastNode)
	resetLayers := getResetGradsNodeLayers(lastNode)

	parallel := 8

	wg := &sync.WaitGroup{}

	fChan := make(chan *Data)
	bChan := make(chan *Data)
	rChan := make(chan *Data)

	for i := 0; i < parallel; i++ {
		go func() {
			for {
				select {
				case node := <-fChan:
					node.calcData()
				case node := <-bChan:
					node.calcGrad()
				case node := <-rChan:
					node.Grad.Fill(0.0)
				}
				wg.Done()
			}
		}()
	}

	return &Pipeline{
		wg: wg,

		forwardChan:   fChan,
		forwardLayers: forwardLayers,

		backwardChan:   bChan,
		backwardLayers: backwardLayers,

		resetChan:   rChan,
		resetLayers: resetLayers,
	}
}

type Pipeline struct {
	wg *sync.WaitGroup

	forwardChan   chan *Data
	forwardLayers NodeLayers

	backwardChan   chan *Data
	backwardLayers NodeLayers

	resetChan   chan *Data
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
			p.forwardChan <- node
		}
		p.wg.Wait()
	}
}

func (p *Pipeline) Backward() {
	p.resetLayers[0][0].Grad.Fill(1.0)
	for _, nodes := range p.resetLayers[1:] {
		if len(nodes) == 1 {
			nodes[0].Grad.Fill(0.0)
			continue
		}

		p.wg.Add(len(nodes))
		for _, node := range nodes {
			p.resetChan <- node
		}
		p.wg.Wait()
	}

	for _, nodes := range p.backwardLayers {
		if len(nodes) == 1 {
			nodes[0].calcGrad()
			continue
		}

		p.wg.Add(len(nodes))
		for _, node := range nodes {
			p.backwardChan <- node
		}
		p.wg.Wait()
	}
}
