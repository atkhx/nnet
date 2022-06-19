package activation

import (
	"sync"

	"github.com/atkhx/nnet/data"
)

func New(f Activation, options ...Option) *layer {
	layer := &layer{activation: f}
	applyOptions(layer, options...)

	return layer
}

type layer struct {
	iWidth, iHeight, iDepth int

	activation Activation

	inputs     *data.Data
	output     *data.Data
	deltas     *data.Data
	gradInputs *data.Data

	threads        int
	activateInChan chan int
	backpropInChan chan int
	wg             sync.WaitGroup
}

func (l *layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.output = &data.Data{}
	l.output.InitCube(w, h, d)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(w, h, d)

	if l.threads == 0 {
		l.threads = len(l.output.Data)
	}

	l.activateInChan = make(chan int, l.threads)
	l.backpropInChan = make(chan int, l.threads)

	for i := 0; i < l.threads; i++ {
		go func() {
			for {
				select {
				case fi := <-l.activateInChan:
					l.activateFilter(fi)
				case fi := <-l.backpropInChan:
					l.backpropFilter(fi)
				}
				l.wg.Done()
			}
		}()
	}

	return w, h, d
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	l.wg.Add(l.iDepth)
	for i := 0; i < l.iDepth; i++ {
		l.activateInChan <- i
	}
	l.wg.Wait()

	return l.output
}

func (l *layer) activateFilter(z int) {
	iSquare := l.iWidth * l.iHeight
	for i := z * iSquare; i < (z+1)*iSquare; i++ {
		l.output.Data[i] = l.activation.Forward(l.inputs.Data[i])
	}
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.deltas = deltas

	l.wg.Add(l.iDepth)
	for i := 0; i < l.iDepth; i++ {
		l.backpropInChan <- i
	}
	l.wg.Wait()
	return l.gradInputs
}

func (l *layer) backpropFilter(z int) {
	iSquare := l.iWidth * l.iHeight
	for i := z * iSquare; i < (z+1)*iSquare; i++ {
		l.gradInputs.Data[i] = l.deltas.Data[i] * l.activation.Backward(l.output.Data[i])
	}
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
