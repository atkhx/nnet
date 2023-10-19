package num

type MMConfig struct {
	Alpha float32
}

type MMOption func(*MMConfig)

func WithMatrixMultiplyAlpha(alpha float32) MMOption {
	return func(config *MMConfig) {
		config.Alpha = alpha
	}
}
