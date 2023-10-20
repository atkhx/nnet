package metal

import (
	"context"

	"github.com/atkhx/nnet/num"
)

type NodeLayers []num.Nodes

func getDistinctNodes(aData *num.Data) num.Nodes {
	nodes := num.Nodes{}
	added := map[*num.Data]struct{}{}

	var addFunc func(node *num.Data)

	addFunc = func(node *num.Data) {
		for _, srcNode := range node.SrcNodes {
			if _, ok := added[srcNode]; !ok {
				addFunc(srcNode)
			}
		}

		if _, ok := added[node]; !ok {
			added[node] = struct{}{}
			nodes = append(nodes, node)
		}
	}

	addFunc(aData)
	return nodes
}

func getForwardNodeList(aData *num.Data) num.Nodes {
	distinctNodes := getDistinctNodes(aData)
	distinctNodesNotAdded := len(distinctNodes)

	nodes := num.Nodes{}
	added := map[*num.Data]struct{}{}

	allSrcNodesWasAdded := func(node *num.Data) bool {
		for _, srcNode := range node.SrcNodes {
			if _, ok := added[srcNode]; !ok {
				return false
			}
		}
		return true
	}

	for distinctNodesNotAdded > 0 {
		for _, node := range distinctNodes {
			if _, ok := added[node]; ok {
				continue
			}

			if allSrcNodesWasAdded(node) {
				nodes = append(nodes, node)
				added[node] = struct{}{}
				distinctNodesNotAdded--
				break
			}

			panic("no processable nodes")
		}
	}

	return nodes
}

func getNodeLayers(aData *num.Data, skipNodeFn func(*num.Data) bool) NodeLayers {
	nodeList := getForwardNodeList(aData)
	notAdded := len(nodeList)

	nodeLayers := NodeLayers{}
	nodeLevel := map[*num.Data]int{}

	getMaxSrcNodeLevel := func(srcNodes num.Nodes) int {
		result := -1
		for _, node := range srcNodes {
			if nodeLevel[node] > result {
				result = nodeLevel[node]
			}
		}
		return result
	}

	for notAdded > 0 {
		for _, node := range nodeList {
			if _, nodeProcessed := nodeLevel[node]; nodeProcessed {
				continue
			}
			notAdded--

			if skipNodeFn != nil && skipNodeFn(node) {
				nodeLevel[node] = -1
				continue
			}

			level := getMaxSrcNodeLevel(node.SrcNodes) + 1
			if len(nodeLayers) == level {
				nodeLayers = append(nodeLayers, num.Nodes{})
			}

			nodeLayers[level] = append(nodeLayers[level], node)
			nodeLevel[node] = level
		}
	}
	return nodeLayers
}

func getForwardNodeLayers(aData *num.Data) NodeLayers {
	return getNodeLayers(aData, func(node *num.Data) bool {
		return node.CalcData == nil
	})
}

func getBackwardNodeLayers(aData *num.Data) NodeLayers {
	nodes := getNodeLayers(aData, func(node *num.Data) bool {
		return node.CalcGrad == nil
	})

	bwdNodes := make(NodeLayers, 0, len(nodes))
	for i := len(nodes); i > 0; i-- {
		bwdNodes = append(bwdNodes, nodes[i-1])
	}
	return bwdNodes
}

func getResetGradsNodeLayers(aData *num.Data) NodeLayers {
	nodes := getNodeLayers(aData, nil)
	bwdNodes := make(NodeLayers, 0, len(nodes))
	for i := len(nodes); i > 0; i-- {
		bwdNodes = append(bwdNodes, nodes[i-1])
	}
	return bwdNodes
}

func getResetGradsNodeFuncs(aData *num.Data) []func(ctx context.Context) {
	nodes := getDistinctNodes(aData)
	var result []func(ctx context.Context)
	for i := len(nodes); i > 0; i-- {
		i := i
		if nodes[i-1].SkipResetGrad {
			continue
		}
		if i == len(nodes) {
			result = append(result, func(ctx context.Context) {
				//cbuf := mps.CommandBufferFromContext(ctx)
				//cbuf.FillMTLBuffer(nodes[i-1].gradBuffer, 1.0)
				Float32s(nodes[i-1].Grad).Ones()
			})
		} else {
			result = append(result, func(ctx context.Context) {
				//cbuf := mps.CommandBufferFromContext(ctx)
				//cbuf.ClearMTLBuffer(nodes[i-1].gradBuffer)
				Float32s(nodes[i-1].Grad).Zero()
			})
		}
	}
	return result
}
