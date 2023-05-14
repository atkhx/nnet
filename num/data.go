package num

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

func (input *Data) Copy() *Data {
	return &Data{
		Data: make(Float64s, len(input.Data)),
		Grad: make(Float64s, len(input.Data)),
		Dims: input.Dims,

		srcNodes: Nodes{input},
	}
}

func (input *Data) Forward() {
	input.calcData()
}

func (input *Data) Backward() {
	input.calcGrad()
}

func (input *Data) ResetGrads(v float64) {
	input.resetGrads(v)
}

func (input *Data) resetGrads(v float64) {
	input.Grad.Fill(v)
	input.srcNodes.Each(func(node *Data) {
		node.resetGrads(0)
	})
}

func (input *Data) StringData() string {
	return input.Data.String(input.Dims)
}

func (input *Data) StringGrad() string {
	return input.Grad.String(input.Dims)
}
