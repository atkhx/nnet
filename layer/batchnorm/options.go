package batchnorm

func applyOptions(layer *BatchNorm, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	WithInputDepth(1),
}

type Option func(layer *BatchNorm)

func WithInputSize(inputSize int) Option {
	return func(layer *BatchNorm) {
		layer.inputSize = inputSize
	}
}

func WithInputDepth(inputDepth int) Option {
	return func(layer *BatchNorm) {
		layer.inputDepth = inputDepth
	}
}
