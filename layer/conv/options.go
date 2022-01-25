package conv

type Option func(layer *layer)

const (
	defaultFilterWidth  = 3
	defaultFilterHeight = 3
	defaultFiltersCount = 1
	defaultStride       = 1
	defaultPadding      = 0
)

func defaults(layer *layer) {
	layer.FWidth = defaultFilterWidth
	layer.FHeight = defaultFilterHeight
	layer.FCount = defaultFiltersCount
	layer.FStride = defaultStride
	layer.FPadding = defaultPadding
}

func FilterSize(size int) Option {
	return func(layer *layer) {
		layer.FWidth = size
		layer.FHeight = size
	}
}

func FiltersCount(count int) Option {
	return func(layer *layer) {
		layer.FCount = count
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
