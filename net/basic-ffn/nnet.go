package basic_ffn

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/layer"

	"log"
)

type Layers []layer.Layer

func New(iWidth, iHeight, iDepth int, layers Layers) *ffnet {
	return &ffnet{
		iWidth:  iWidth,
		iHeight: iHeight,
		iDepth:  iDepth,
		Layers:  layers,
	}
}

type ffnet struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	Layers Layers
}

func (n *ffnet) Init() (err error) {
	w, h, d := n.iWidth, n.iHeight, n.iDepth

	log.Printf("input [*]: %d:%d:%d, %T", w, h, d, n)
	for i := 0; i < len(n.Layers); i++ {
		w, h, d = n.Layers[i].InitDataSizes(w, h, d)
		log.Printf("layer [%d]: %d:%d:%d, %T", i, w, h, d, n.Layers[i])
	}

	n.oWidth, n.oHeight, n.oDepth = w, h, d

	return
}

func (n *ffnet) Activate(inputs *data.Data) *data.Data {
	for i := 0; i < len(n.Layers); i++ {
		inputs = n.Layers[i].Activate(inputs)
	}
	return inputs
}

func (n *ffnet) Backprop(deltas *data.Data) (gradient *data.Data) {
	gradient = deltas.Copy()

	for i := len(n.Layers) - 1; i >= 0; i-- {
		gradient = n.Layers[i].Backprop(gradient)
	}
	return gradient
}

func (n *ffnet) GetLayersCount() int {
	return len(n.Layers)
}

func (n *ffnet) GetLayer(index int) layer.Layer {
	if index > -1 && index < len(n.Layers) {
		return n.Layers[index]
	}
	return nil
}
