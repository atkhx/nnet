package net

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

type Layers []nnet.Layer

func New(layers Layers) *FeedForward {
	return &FeedForward{
		Layers: layers,
	}
}

type FeedForward struct {
	Layers Layers
}

func (n *FeedForward) Forward(inputs *data.Data) *data.Data {
	for i := 0; i < len(n.Layers); i++ {
		inputs = n.Layers[i].Forward(inputs)
	}
	return inputs
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
