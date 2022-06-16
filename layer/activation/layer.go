//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package activation

import (
	"sync"

	"github.com/atkhx/nnet/data"
)

type ActivationFunc interface {
	Forward(v float64) float64
	Backward(v float64) float64
}

func New(f ActivationFunc, options ...Option) *layer {
	layer := &layer{Activation: f}

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

type layer struct {
	iWidth, iHeight, iDepth int

	inputs *data.Data
	output *data.Data

	gradInputs *data.Data
	Activation ActivationFunc

	Threads int

	deltas *data.Data

	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.output = &data.Data{}
	l.output.InitCube(w, h, d)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(w, h, d)

	if l.Threads == 0 {
		l.Threads = len(l.output.Data)
	}

	l.activateInChan = make(chan int, l.Threads)
	l.backpropInChan = make(chan int, l.Threads)

	for i := 0; i < l.Threads; i++ {
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
		l.output.Data[i] = l.Activation.Forward(l.inputs.Data[i])
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
		l.gradInputs.Data[i] = l.deltas.Data[i] * l.Activation.Backward(l.output.Data[i])
	}
}

func (l *layer) Activate2(inputs *data.Data) *data.Data {
	l.inputs = inputs

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = l.Activation.Forward(l.inputs.Data[i])
	}

	return l.output
}

func (l *layer) Backprop2(deltas *data.Data) *data.Data {
	for i := 0; i < len(l.gradInputs.Data); i++ {
		l.gradInputs.Data[i] = deltas.Data[i] * l.Activation.Backward(l.output.Data[i])
	}
	return l.gradInputs
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
