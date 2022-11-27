package conv

func applyOptions(layer *Layer, options ...Option) {
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
	IsTrainable(true),
}

type Option func(layer *Layer)

type InitWeightsParams struct {
	WeightMinThreshold float64
	WeightMaxThreshold float64
	BiasInitialValue   float64
}

func FilterSize(size int) Option {
	return func(layer *Layer) {
		layer.FWidth = size
		layer.FHeight = size
	}
}

func FiltersCount(count int) Option {
	return func(layer *Layer) {
		layer.FCount = count
	}
}

func Padding(padding int) Option {
	return func(layer *Layer) {
		layer.FPadding = padding
	}
}

func Stride(stride int) Option {
	return func(layer *Layer) {
		layer.FStride = stride
	}
}

func InitWeights(value InitWeightsParams) Option {
	return func(layer *Layer) {
		layer.initWeights = value
	}
}

func IsTrainable(trainable bool) Option {
	return func(layer *Layer) {
		layer.Trainable = trainable
	}
}
