package conv

func applyOptions(layer *Conv, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

//nolint:gomnd
var defaults = []Option{
	FilterSize(3),
	FiltersCount(1),
	Stride(1),
	Padding(0),
	InitWeights(InitWeightsParams{-0.7, 0.7, 0.1}),
	IsTrainable(true),
}

type Option func(layer *Conv)

type InitWeightsParams struct {
	WeightMinThreshold float64
	WeightMaxThreshold float64
	BiasInitialValue   float64
}

func FilterSize(size int) Option {
	return func(layer *Conv) {
		layer.FWidth = size
		layer.FHeight = size
	}
}

func FiltersCount(count int) func(layer *Conv) {
	return func(layer *Conv) {
		layer.FCount = count
	}
}

func Padding(padding int) func(layer *Conv) {
	return func(layer *Conv) {
		layer.FPadding = padding
	}
}

func Stride(stride int) func(layer *Conv) {
	return func(layer *Conv) {
		layer.FStride = stride
	}
}

func InitWeights(value InitWeightsParams) func(layer *Conv) {
	return func(layer *Conv) {
		layer.InitWeightsParams = value
	}
}

func IsTrainable(trainable bool) func(layer *Conv) {
	return func(layer *Conv) {
		layer.Trainable = trainable
	}
}
