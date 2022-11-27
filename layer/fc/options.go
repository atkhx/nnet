package fc

func applyOptions(layer *FC, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

var defaults = []Option{
	OutputSizes(1, 1, 1),
	IsTrainable(true),
}

type Option func(layer *FC)

func OutputSizes(w, h, d int) func(layer *FC) {
	return func(layer *FC) {
		layer.OWidth = w
		layer.OHeight = h
		layer.ODepth = d
	}
}

func IsTrainable(trainable bool) func(layer *FC) {
	return func(layer *FC) {
		layer.Trainable = trainable
	}
}
