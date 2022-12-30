package net

import (
	"log"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

type Layers []nnet.Layer

func New(iWidth, iHeight, iDepth int, layers Layers) *FeedForward {
	return &FeedForward{
		IWidth:  iWidth,
		IHeight: iHeight,
		IDepth:  iDepth,
		Layers:  layers,
	}
}

type FeedForward struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	Layers Layers
}

func (n *FeedForward) Init() (err error) {
	w, h, d := n.IWidth, n.IHeight, n.IDepth

	log.Printf("input [*]: %d:%d:%d, %T", w, h, d, n)
	for i := 0; i < len(n.Layers); i++ {
		w, h, d = n.Layers[i].InitDataSizes(w, h, d)
		log.Printf("layer [%d]: %d:%d:%d, %T", i, w, h, d, n.Layers[i])
	}

	n.OWidth, n.OHeight, n.ODepth = w, h, d

	return
}

func (n *FeedForward) Forward(inputs *data.Data) *data.Data {
	for i := 0; i < len(n.Layers); i++ {
		inputs = n.Layers[i].Forward(inputs)
	}
	return inputs
}

func (n *FeedForward) Backward(deltas *data.Data) (gradient *data.Data) {
	gradient = deltas.Copy()

	for i := len(n.Layers) - 1; i >= 0; i-- {
		gradient = n.Layers[i].Backward(gradient)
	}
	return gradient
}

func (n *FeedForward) GetLayersCount() int {
	return len(n.Layers)
}

func (n *FeedForward) GetLayer(index int) nnet.Layer {
	if index > -1 && index < len(n.Layers) {
		return n.Layers[index]
	}
	return nil
}
