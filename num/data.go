package num

type Nodes []*Data
type Data struct {
	Data Float64s
	Grad Float64s `json:"-"`
	Dims Dims

	srcNodes Nodes
	calcData func()
	calcGrad func()
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

func (aData *Data) Copy() *Data {
	return &Data{
		Data: make(Float64s, len(aData.Data)),
		Grad: make(Float64s, len(aData.Data)),
		Dims: aData.Dims,

		srcNodes: Nodes{aData},
	}
}

func (aData *Data) StringData() string {
	return aData.Data.String(aData.Dims)
}

func (aData *Data) StringGrad() string {
	return aData.Grad.String(aData.Dims)
}

func (aData *Data) Forward() {
	aData.calcData()
}
