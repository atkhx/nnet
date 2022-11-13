package pooling

func applyOptions(layer *Layer, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	FilterSize(2),
	Stride(2),
	Padding(0),
}

type Option func(layer *Layer)

func FilterSize(size int) Option {
	return func(layer *Layer) {
		layer.fWidth = size
		layer.fHeight = size
	}
}

func Padding(padding int) Option {
	return func(layer *Layer) {
		layer.fPadding = padding
	}
}

func Stride(stride int) Option {
	return func(layer *Layer) {
		layer.fStride = stride
	}
}
