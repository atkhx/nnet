package fc

func applyOptions(layer *layer, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	OutputSizes(1, 1, 1),
}

type Option func(layer *layer)

func OutputSizes(w, h, d int) Option {
	return func(layer *layer) {
		layer.oWidth = w
		layer.oHeight = h
		layer.oDepth = d
	}
}

func Threads(threads int) Option {
	return func(layer *layer) {
		layer.threads = threads
	}
}
