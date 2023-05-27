package num

import (
	"fmt"
)

type Nodes []*Data

func (nodes Nodes) Each(fn func(node *Data)) {
	for _, node := range nodes {
		fn(node)
	}
}

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

func (aData *Data) Forward() {
	aData.calcData()
}

func (aData *Data) Backward() {
	aData.calcGrad()
}

func (aData *Data) ResetGrads(v float64) {
	var resetGrads func(node *Data, v float64)

	visitedNodes := map[*Data]struct{}{}
	resetGrads = func(node *Data, v float64) {
		visitedNodes[node] = struct{}{}
		node.Grad.Fill(v)
		for _, srcNode := range node.srcNodes {
			if _, ok := visitedNodes[srcNode]; !ok {
				resetGrads(srcNode, 0)
			}
		}
	}

	resetGrads(aData, v)
	//aData.checkGradsIsEmpty(v)
	//os.Exit(1)
}

func (aData *Data) checkGradsIsEmpty(v float64) {
	var checkGrads func(node *Data, v float64)
	checkGrads = func(node *Data, v float64) {
		for _, val := range node.Grad {
			if val != v {
				panic(fmt.Errorf("node grad is not empty, expected %v, actual %v", v, val))
			}
		}
		for _, srcNode := range node.srcNodes {
			checkGrads(srcNode, 0)
		}
	}

	checkGrads(aData, v)
}

func (aData *Data) StringData() string {
	return aData.Data.String(aData.Dims)
}

func (aData *Data) StringGrad() string {
	return aData.Grad.String(aData.Dims)
}
