package pooling

func applyOptions(layer *layer, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	FilterSize(2),
	Stride(2),
	Padding(0),
}

type Option func(layer *layer)

func FilterSize(size int) Option {
	return func(layer *layer) {
		layer.fWidth = size
		layer.fHeight = size
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
