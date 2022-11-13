package fc

func applyOptions(layer *Layer, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	OutputSizes(1, 1, 1),
	IsTrainable(true),
}

type Option func(layer *Layer)

func OutputSizes(w, h, d int) Option {
	return func(layer *Layer) {
		layer.oWidth = w
		layer.oHeight = h
		layer.oDepth = d
	}
}

func IsTrainable(trainable bool) Option {
	return func(layer *Layer) {
		layer.Trainable = trainable
	}
}
