package pooling

type Option func(layer *layer)

func defaults(layer *layer) {
	layer.FWidth = 2
	layer.FHeight = 2
	layer.FStride = 2
	layer.FPadding = 0
}

func FilterSize(size int) Option {
	return func(layer *layer) {
		layer.FWidth = size
		layer.FHeight = size
	}
}

func Padding(padding int) Option {
	return func(layer *layer) {
		layer.FPadding = padding
	}
}

func Stride(stride int) Option {
	return func(layer *layer) {
		layer.FStride = stride
	}
}
