package activation

func applyOptions(layer *layer, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

type Option func(layer *layer)

func Threads(threads int) Option {
	return func(layer *layer) {
		layer.threads = threads
	}
}
