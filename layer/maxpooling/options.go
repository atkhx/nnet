package maxpooling

func applyOptions(layer *MaxPool, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	FilterSize(2),
	Stride(2),
	Padding(0),
}

type Option func(layer *MaxPool)

func FilterSize(size int) Option {
	return func(layer *MaxPool) {
		layer.FWidth = size
		layer.FHeight = size
	}
}

func Padding(padding int) Option {
	return func(layer *MaxPool) {
		layer.FPadding = padding
	}
}

func Stride(stride int) Option {
	return func(layer *MaxPool) {
		layer.FStride = stride
	}
}
