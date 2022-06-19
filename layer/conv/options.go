package conv

func applyOptions(layer *layer, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	FilterSize(3),
	FiltersCount(1),
	Stride(1),
	Padding(0),
	InitWeights(InitWeightsParams{-0.7, 0.7, 0.1}),
}

type Option func(layer *layer)

type InitWeightsParams struct {
	WeightMinThreshold float64
	WeightMaxThreshold float64
	BiasInitialValue   float64
}

func FilterSize(size int) Option {
	return func(layer *layer) {
		layer.FWidth = size
		layer.FHeight = size
	}
}

func FiltersCount(count int) Option {
	return func(layer *layer) {
		layer.fCount = count
	}
}

func Padding(padding int) Option {
	return func(layer *layer) {
		layer.fPadding = padding
	}
}

func Stride(stride int) Option {
	return func(layer *layer) {
		layer.fStride = stride
	}
}

func Threads(threads int) Option {
	return func(layer *layer) {
		layer.threads = threads
	}
}

func InitWeights(value InitWeightsParams) Option {
	return func(layer *layer) {
		layer.InitWeights = value
	}
}
