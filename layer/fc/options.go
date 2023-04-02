package fc

func applyOptions(layer *FC, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{}

type Option func(layer *FC)

func WithLayerSize(layerSize int) Option {
	return func(layer *FC) {
		layer.layerSize = layerSize
	}
}

func WithInputSize(inputSize int) Option {
	return func(layer *FC) {
		layer.inputSize = inputSize
	}
}

func WithBatchSize(batchSize int) Option {
	return func(layer *FC) {
		layer.batchSize = batchSize
	}
}

func WithGain(gain float64) Option {
	return func(layer *FC) {
		layer.gain = gain
	}
}

func WithBiases(WithBiases bool) Option {
	return func(layer *FC) {
		layer.WithBiases = WithBiases
	}
}
