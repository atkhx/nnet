package num

import "fmt"

func New(dims Dims, srcNodes ...*Data) *Data {
	return &Data{
		Data: make(Float64s, dims.Size()),
		Grad: make(Float64s, dims.Size()),
		Dims: dims,

		srcNodes: srcNodes,
	}
}

func NewWithValues(dims Dims, values Float64s, srcNodes ...*Data) *Data {
	if len(values) != dims.Size() {
		panic("invalid values size")
	}

	return &Data{
		Data: values,
		Grad: make(Float64s, dims.Size()),
		Dims: dims,

		srcNodes: srcNodes,
	}
}

func NewRandNorm(dims Dims) *Data {
	return &Data{
		Data: NewRandNormFloat64s(dims.Size()),
		Grad: NewFloat64s(dims.Size()),
		Dims: dims,
	}
}

func NewRandNormWeighted(dims Dims, w float64) *Data {
	return &Data{
		Data: NewRandNormWeightedFloat64s(dims.Size(), w),
		Grad: NewFloat64s(dims.Size()),
		Dims: dims,
	}
}

type Data struct {
	Data Float64s
	Grad Float64s `json:"-"`
	Dims Dims
	Name string

	srcNodes Nodes
	calcData func()
	calcGrad func()
}

func (aData *Data) Copy() *Data {
	return &Data{
		Data: make(Float64s, len(aData.Data)),
		Grad: make(Float64s, len(aData.Data)),
		Dims: aData.Dims,

		srcNodes: Nodes{aData},
	}
}

func (aData *Data) getDistinctNodes() Nodes {
	nodes := Nodes{}
	added := map[*Data]struct{}{}

	var addFunc func(node *Data)

	addFunc = func(node *Data) {
		for _, srcNode := range node.srcNodes {
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

func (aData *Data) getForwardGraph() Nodes {
	distinctNodes := aData.getDistinctNodes()
	distinctNodesNotAdded := len(distinctNodes)

	nodes := Nodes{}
	added := map[*Data]struct{}{}

	allSrcNodesWasAdded := func(node *Data) bool {
		for _, srcNode := range node.srcNodes {
			if _, ok := added[srcNode]; !ok {
				return false
			}
		}
		return true
	}

	for distinctNodesNotAdded > 0 {
		shouldPanic := true
		for i := 0; i < len(distinctNodes); i++ {
			node := distinctNodes[i]
			if _, ok := added[node]; ok {
				continue
			}

			if allSrcNodesWasAdded(node) {
				nodes = append(nodes, node)
				added[node] = struct{}{}
				distinctNodesNotAdded--
				shouldPanic = false
				break
			}
		}

		if shouldPanic {
			panic("graph build failed")
		}
	}

	return nodes
}

func (aData *Data) GetForwardNodes() Nodes {
	nodes := aData.getForwardGraph()
	fwdNodes := make(Nodes, 0, len(nodes))
	for _, node := range nodes {
		if node.calcData != nil {
			fwdNodes = append(fwdNodes, node)
		}
	}
	return fwdNodes
}

func (aData *Data) GetForwardNodesLayers() NodesLayers {
	// nodes -> input, weight1, weight2, output1, output2, outputsSum
	nodes := aData.getForwardGraph()
	fwdNodes := NodesLayers{}
	addedByLevels := map[*Data]int{}

	getMaxSrcNodeLevel := func(nodes Nodes) int {
		result := -1
		for _, node := range nodes {
			if _, ok := addedByLevels[node]; !ok {
				if node.calcData != nil {
					panic("not added node has calcData")
				}
				if len(node.srcNodes) > 0 {
					panic("not added node has srcNodes")
				}
				continue
			}

			if addedByLevels[node] > result {
				result = addedByLevels[node]
			}
		}

		return result
	}

	freeNodes := len(nodes)
	for freeNodes > 0 {
		shouldPanic := true
		for _, node := range nodes {
			if _, ok := addedByLevels[node]; ok {
				continue
			}

			if node.calcData == nil {
				freeNodes--
				shouldPanic = false
				continue
			}

			maxSrcNodeLevel := getMaxSrcNodeLevel(node.srcNodes)
			currentNodeLevel := maxSrcNodeLevel + 1

			if len(fwdNodes) == currentNodeLevel {
				fwdNodes = append(fwdNodes, Nodes{})
			}

			fwdNodes[currentNodeLevel] = append(fwdNodes[currentNodeLevel], node)

			addedByLevels[node] = currentNodeLevel
			freeNodes--
			shouldPanic = false
		}

		if shouldPanic {
			panic("build leveled nodes graph failed")
		}
	}

	fmt.Println("nodes layers")
	forwardNodesTotal := 0
	for i, nodes := range fwdNodes {
		seen := map[*Data]struct{}{}
		dups := 0
		for _, node := range nodes {
			if _, ok := seen[node]; !ok {
				seen[node] = struct{}{}
			} else {
				dups++
			}
		}
		if dups > 0 {
			panic("duplicates!")
		}
		forwardNodesTotal += len(nodes)
		fmt.Println("layer", i, "nodes count:", len(nodes), "dups:", dups, nodes.String())
	}
	fmt.Println()

	// forwardNodes total 323
	fmt.Println("forwardNodes total", forwardNodesTotal)

	return fwdNodes
}

func (aData *Data) GetBackwardNodes() Nodes {
	nodes := aData.getForwardGraph()
	revNodes := make(Nodes, 0, len(nodes))
	for i := len(nodes); i > 0; i-- {
		if nodes[i-1].calcGrad != nil {
			revNodes = append(revNodes, nodes[i-1])
		}
	}
	return revNodes
}

func (aData *Data) GetBackwardNodesLayers() NodesLayers {
	// nodes -> input, weight1, weight2, output1, output2, outputsSum
	nodes := aData.getForwardGraph()
	fwdNodes := NodesLayers{}
	addedByLevels := map[*Data]int{}

	getMaxSrcNodeLevel := func(nodes Nodes) int {
		result := -1
		for _, node := range nodes {
			if _, ok := addedByLevels[node]; !ok {
				if node.calcGrad != nil {
					panic("not added node has calcGrad")
				}
				if len(node.srcNodes) > 0 {
					panic("not added node has srcNodes")
				}
				continue
			}

			if addedByLevels[node] > result {
				result = addedByLevels[node]
			}
		}

		return result
	}

	freeNodes := len(nodes)
	for freeNodes > 0 {
		shouldPanic := true
		for _, node := range nodes {
			if _, ok := addedByLevels[node]; ok {
				continue
			}

			if node.calcGrad == nil {
				freeNodes--
				shouldPanic = false
				continue
			}

			maxSrcNodeLevel := getMaxSrcNodeLevel(node.srcNodes)
			currentNodeLevel := maxSrcNodeLevel + 1

			if len(fwdNodes) == currentNodeLevel {
				fwdNodes = append(fwdNodes, Nodes{})
			}

			fwdNodes[currentNodeLevel] = append(fwdNodes[currentNodeLevel], node)

			addedByLevels[node] = currentNodeLevel
			freeNodes--
			shouldPanic = false
		}

		if shouldPanic {
			panic("build leveled nodes graph failed")
		}
	}

	fmt.Println("nodes layers")
	forwardNodesTotal := 0
	for i, nodes := range fwdNodes {
		seen := map[*Data]struct{}{}
		dups := 0
		for _, node := range nodes {
			if _, ok := seen[node]; !ok {
				seen[node] = struct{}{}
			} else {
				dups++
			}
		}
		if dups > 0 {
			panic("duplicates!")
		}
		forwardNodesTotal += len(nodes)
		fmt.Println("layer", i, "nodes count:", len(nodes), "dups:", dups, nodes.String())
	}
	fmt.Println()

	// forwardNodes total 323
	fmt.Println("backwardNodes total", forwardNodesTotal)

	return fwdNodes
}

func (aData *Data) GetResetGradsNodes() Nodes {
	nodes := aData.getForwardGraph()
	revNodes := make(Nodes, 0, len(nodes))
	for i := len(nodes); i > 0; i-- {
		revNodes = append(revNodes, nodes[i-1])
	}
	return revNodes
}

func (aData *Data) StringData() string {
	return aData.Data.String(aData.Dims)
}

func (aData *Data) StringGrad() string {
	return aData.Grad.String(aData.Dims)
}
