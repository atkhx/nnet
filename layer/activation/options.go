package activation

type Option func(layer *layer)

func Threads(threads int) Option {
	return func(layer *layer) {
		layer.Threads = threads
	}
}
