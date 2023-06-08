package num

import (
	"strings"
	"sync"
)

type Nodes []*Data

func (nodes Nodes) String() string {
	names := make([]string, 0, len(nodes))
	for _, node := range nodes {
		names = append(names, node.Name)
	}
	return strings.Join(names, "\n")
}

func (nodes Nodes) Each(fn func(node *Data)) {
	for _, node := range nodes {
		fn(node)
	}
}

func (nodes Nodes) Forward() {
	for _, node := range nodes {
		node.calcData()
	}
}

func (nodes Nodes) Backward() {
	for _, node := range nodes {
		node.calcGrad()
	}
}

func (nodes Nodes) ResetGrad() {
	nodes[0].Grad.Fill(1)
	for _, node := range nodes[1:] {
		node.Grad.Fill(0)
	}
}

type NodesLayers []Nodes

func (layers NodesLayers) Forward() {
	wg := sync.WaitGroup{}
	for _, nodes := range layers {
		switch len(nodes) {
		case 0:
			panic("layer without nodes")
		case 1:
			nodes[0].calcData()
		default:
			wg.Add(len(nodes))
			for _, node := range nodes {
				go func(node *Data) {
					node.calcData()
					wg.Done()
				}(node)
			}
			wg.Wait()
		}
	}
}

func (layers NodesLayers) Backward() {
	wg := sync.WaitGroup{}
	for i := len(layers); i > 0; i-- {
		nodes := layers[i-1]
		switch len(nodes) {
		case 0:
			panic("layer without nodes")
		case 1:
			nodes[0].calcGrad()
		default:
			wg.Add(len(nodes))
			for _, node := range nodes {
				go func(node *Data) {
					node.calcGrad()
					wg.Done()
				}(node)
			}
			wg.Wait()
		}
	}
}
