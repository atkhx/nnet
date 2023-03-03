package conv

func applyOptions(layer *Conv, options ...Option) {
	for _, opt := range options {
		opt(layer)
	}
}

//nolint:gomnd
var defaults = []Option{
	WithStride(1),
	WithPadding(0),
}

type Option func(layer *Conv)

func WithInputSize(width, height, channels int) Option {
	return func(layer *Conv) {
		layer.inputWidth = width
		layer.inputHeight = height
		layer.inputChannels = channels
	}
}

func WithFilterSize(filterSize int) Option {
	return func(layer *Conv) {
		layer.FilterSize = filterSize
	}
}

func WithPadding(padding int) Option {
	return func(layer *Conv) {
		layer.Padding = padding
	}
}

func WithStride(stride int) Option {
	return func(layer *Conv) {
		layer.Stride = stride
	}
}

func WithFiltersCount(filtersCount int) Option {
	return func(layer *Conv) {
		layer.FiltersCount = filtersCount
	}
}
