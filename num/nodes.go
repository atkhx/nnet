package num

type Nodes []*Data

type NodeLayers []Nodes

func getDistinctNodes(aData *Data) Nodes {
	nodes := Nodes{}
	added := map[*Data]struct{}{}

	var addFunc func(node *Data)

	addFunc = func(node *Data) {
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

func getForwardNodeList(aData *Data) Nodes {
	distinctNodes := getDistinctNodes(aData)
	distinctNodesNotAdded := len(distinctNodes)

	nodes := Nodes{}
	added := map[*Data]struct{}{}

	allSrcNodesWasAdded := func(node *Data) bool {
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

func getNodeLayers(aData *Data, skipNodeFn func(*Data) bool) NodeLayers {
	nodeList := getForwardNodeList(aData)
	notAdded := len(nodeList)

	nodeLayers := NodeLayers{}
	nodeLevel := map[*Data]int{}

	getMaxSrcNodeLevel := func(srcNodes Nodes) int {
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
				nodeLayers = append(nodeLayers, Nodes{})
			}

			nodeLayers[level] = append(nodeLayers[level], node)
			nodeLevel[node] = level
		}
	}
	return nodeLayers
}

func GetForwardNodeLayers(aData *Data) NodeLayers {
	return getNodeLayers(aData, func(node *Data) bool {
		return node.CalcData == nil
	})
}

func GetBackwardNodeLayers(aData *Data) NodeLayers {
	nodes := getNodeLayers(aData, func(node *Data) bool {
		return node.CalcGrad == nil
	})

	bwdNodes := make(NodeLayers, 0, len(nodes))
	for i := len(nodes); i > 0; i-- {
		bwdNodes = append(bwdNodes, nodes[i-1])
	}
	return bwdNodes
}

func GetResetGradsNodes(aData *Data) Nodes {
	nodes := getDistinctNodes(aData)

	var result Nodes
	for i := len(nodes); i > 0; i-- {
		if nodes[i-1].SkipResetGrad {
			continue
		}
		result = append(result, nodes[i-1])
	}
	return result
}

//
//func GetResetGradsNodeFuncs(aData *Data) []func(ctx context.Context) {
//	nodes := getDistinctNodes(aData)
//	var result []func(ctx context.Context)
//	for i := len(nodes); i > 0; i-- {
//		i := i
//		if nodes[i-1].SkipResetGrad {
//			continue
//		}
//		if i == len(nodes) {
//			result = append(result, func(ctx context.Context) {
//				blas32.Float32s(nodes[i-1].Grad).Ones()
//			})
//		} else {
//			result = append(result, func(ctx context.Context) {
//				blas32.Float32s(nodes[i-1].Grad).Zero()
//			})
//		}
//	}
//	return result
//}
