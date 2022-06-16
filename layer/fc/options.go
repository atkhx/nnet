package fc

type Option func(layer *layer)

func defaults(layer *layer) {
	layer.OWidth = 1
	layer.OHeight = 1
	layer.ODepth = 1
}

func OutputSizes(w, h, d int) Option {
	return func(layer *layer) {
		layer.OWidth = w
		layer.OHeight = h
		layer.ODepth = d
	}
}

func Threads(threads int) Option {
	return func(layer *layer) {
		layer.Threads = threads
	}
}
